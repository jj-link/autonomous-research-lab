"""Literature search and retrieval over fixed public HTTPS endpoints.

Source content is data, never instructions: records are returned verbatim for
the research record and must never be interpreted as commands to this software.

Retrieval is restricted to public HTTPS text/HTML. URLs carrying credentials
are rejected. Each hostname is resolved once and every resulting address is
vetted against a public-address rule; the connection is pinned to a vetted
address so a later DNS answer cannot redirect an in-flight request, while TLS
continues to verify the original hostname. Every redirect is validated the
same way (at most three), responses are capped at 1 MiB, and stored text is
bounded to 32,000 characters with truncation marked in the record's text.
"""
from __future__ import annotations

import codecs
import email.message
import hashlib
import html.parser
import http.client
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from typing import Any

from .models import LabError

__all__ = [
    "CROSSREF_WORKS_URL",
    "MAX_TEXT_CHARS",
    "RESPONSE_CAP_BYTES",
    "SourceError",
    "SourceTools",
    "bounded_read",
    "crossref_query_url",
    "html_to_text",
    "is_permitted_address",
    "load_strict_json",
    "record_from_crossref_item",
    "redirect_target",
    "resolve_addresses",
    "validate_https_url",
]

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
SEARCH_RESULT_LIMIT = 5
MAX_TEXT_CHARS = 32_000
RESPONSE_CAP_BYTES = 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20.0
MAX_REDIRECTS = 3
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TRUNCATION_MARKER = f"\n[Truncated: stored text is limited to {MAX_TEXT_CHARS} characters]"
_FETCH_HEADERS = {
    "Accept": "text/html, text/plain, application/json;q=0.9, */*;q=0.1",
    "Accept-Encoding": "identity",
    "User-Agent": "autonomous-research-lab/0.1 (source retrieval)",
}


class SourceError(LabError):
    """A source retrieval or search failure with an actionable message."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_https_url(url: str) -> str:
    """Validate a fetch URL and return it without any fragment.

    Only plain https URLs are permitted: no other scheme, no embedded
    credentials, no missing hostname. Query strings are allowed.
    """
    if not isinstance(url, str) or not url.strip():
        raise SourceError("A URL is required")
    stripped = url.strip()
    if any(ord(char) < 33 for char in stripped):
        raise SourceError("URL contains whitespace or control characters")
    try:
        parts = urlsplit(stripped)
        parts.port
    except ValueError as exc:
        raise SourceError(f"Invalid URL authority: {exc}") from exc
    if parts.scheme != "https":
        raise SourceError(f"Only https URLs may be fetched, got {stripped!r}")
    if parts.username or parts.password:
        raise SourceError(f"URLs with embedded credentials are rejected: {stripped!r}")
    if not parts.hostname:
        raise SourceError(f"URL has no hostname: {stripped!r}")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def is_permitted_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for globally routable addresses; loopback, private, link-local,
    multicast, reserved, and unspecified ranges are all refused explicitly
    (``is_global`` alone does not exclude multicast on every Python version)."""
    return (
        address.is_global
        and not address.is_multicast
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_unspecified
        and not address.is_reserved
    )


def resolve_addresses(hostname: str) -> list[str]:
    """Resolve a hostname once and return only vetted public addresses.

    Raises :class:`SourceError` when resolution fails or any returned address is
    not publicly routable (including IPv4-mapped IPv6), so a request can never
    be pointed at internal infrastructure.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise SourceError(f"DNS resolution failed for {hostname}: {exc}") from exc
    addresses: list[str] = []
    for info in infos:
        candidate = ipaddress.ip_address(info[4][0])
        if isinstance(candidate, ipaddress.IPv6Address) and candidate.ipv4_mapped is not None:
            candidate = candidate.ipv4_mapped
        if not is_permitted_address(candidate):
            raise SourceError(f"Refusing to connect to non-public address {candidate} for {hostname}")
        rendered = str(candidate)
        if rendered not in addresses:
            addresses.append(rendered)
    if not addresses:
        raise SourceError(f"No usable addresses were returned for {hostname}")
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pre-vetted address.

    The socket is created against the vetted address (no re-resolution), while
    SNI and certificate verification still use the original hostname.
    """

    def __init__(self, address: str, host: str, port: int | None = None, timeout: float | None = None):
        super().__init__(host, port=port, timeout=timeout)
        self.pinned_address = address

    def connect(self) -> None:
        sock = socket.create_connection((self.pinned_address, self.port), self.timeout, self.source_address)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def bounded_read(stream: Any, cap: int) -> bytes:
    """Read a response body up to ``cap`` bytes; raise SourceError beyond it."""
    data = stream.read(cap + 1)
    if len(data) > cap:
        raise SourceError(f"Response body exceeds the {cap} byte limit")
    return data


def redirect_target(current: str, location: str) -> str:
    """Resolve a Location header against the current URL under the same fetch rules."""
    if not isinstance(location, str) or not location.strip():
        raise SourceError(f"Redirect response from {current} carries no usable Location header")
    return validate_https_url(urljoin(current, location.strip()))


class _TextExtractor(html.parser.HTMLParser):
    """Collects visible text while skipping script/style; block-level tags become newlines."""

    _SKIP = frozenset({"script", "style"})
    _BLOCK = frozenset({
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot",
        "th", "thead", "tr", "ul", "title",
    })

    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self._limit = limit
        self.parts: list[str] = []
        self._length = 0
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.truncated = False

    def _is_block(self, tag: str) -> bool:
        # Namespaced markup such as <jats:p> in abstracts maps to its local name.
        return tag.rpartition(":")[2] in self._BLOCK

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if self._is_block(tag):
            self._append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if self._is_block(tag):
            self._append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self._append(data if data.strip() else " ")

    def _append(self, text: str) -> None:
        if self._length >= self._limit:
            if text.strip():
                self.truncated = True
            return
        room = self._limit - self._length
        if len(text) > room:
            self.parts.append(text[:room])
            self._length = self._limit
            if text[room:].strip():
                self.truncated = True
        else:
            self.parts.append(text)
            self._length += len(text)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def html_to_text(html: str) -> tuple[str, str, bool]:
    """Normalize HTML to bounded visible text; returns (text, title, truncated).

    Scripts and styles are stripped, entities are decoded, block tags become
    newlines, and the result never exceeds MAX_TEXT_CHARS characters; when the
    source contained more text, ``truncated`` is True and the text ends with an
    explicit truncation marker.
    """
    if not isinstance(html, str):
        raise SourceError("HTML must be decoded to a string before normalization")
    extractor = _TextExtractor(MAX_TEXT_CHARS)
    extractor.feed(html)
    extractor.close()
    text = _collapse_whitespace("".join(extractor.parts))
    title = " ".join("".join(extractor.title_parts).split())
    if extractor.truncated:
        text = text[: MAX_TEXT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return text, title, extractor.truncated


def _media_type(content_type: str) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _charset(content_type: str) -> str:
    message = email.message.Message()
    message["Content-Type"] = content_type or "text/plain"
    charset = message.get_param("charset")
    if isinstance(charset, tuple):
        charset = charset[2]
    if not isinstance(charset, str):
        return "utf-8"
    try:
        codecs.lookup(charset)
    except LookupError:
        return "utf-8"
    return charset


def _is_html_media(media: str) -> bool:
    return media in {"text/html", "application/xhtml+xml"}


def _is_supported_media(media: str) -> bool:
    return media.startswith("text/") or media == "application/xhtml+xml"


def source_id(url: str) -> str:
    """Stable identifier derived from the record's URL."""
    return "src_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def build_source_record(url: str, body: bytes, content_type: str, retrieved_at: str | None = None) -> dict:
    """Materialize the persisted record: {id, url, title, text, sha256, retrieved_at}.

    ``sha256`` hashes the raw response body; ``text`` holds the bounded visible
    text (truncation marked in the text when the source contained more).
    """
    media = _media_type(content_type)
    if not _is_supported_media(media):
        raise SourceError(f"Unsupported content type {media!r}; only text and HTML responses are stored")
    decoded = body.decode(_charset(content_type), errors="replace").lstrip("\ufeff")
    title = ""
    truncated = False
    if _is_html_media(media):
        text, title, truncated = html_to_text(decoded)
    elif len(decoded) > MAX_TEXT_CHARS:
        text = decoded[: MAX_TEXT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
        truncated = True
    else:
        text = decoded
    return {
        "id": source_id(url),
        "url": url,
        "title": title,
        "text": text,
        "sha256": hashlib.sha256(body).hexdigest(),
        "retrieved_at": retrieved_at or _now(),
        "truncated": truncated,
        "content_kind": "web_text",
    }


def load_strict_json(text: str) -> Any:
    def reject_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    value = json.loads(text, parse_constant=reject_constant)
    json.dumps(value, allow_nan=False)
    return value


def crossref_query_url(query: str, limit: int) -> str:
    """Fixed public Crossref works endpoint with the query and field selection encoded."""
    return CROSSREF_WORKS_URL + "?" + urlencode({
        "query.bibliographic": query,
        "rows": str(limit),
        "select": "DOI,title,abstract,container-title,issued",
    })


def _strip_markup(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(part for part in value if isinstance(part, str))
    if not isinstance(value, str):
        return ""
    text, _title, _truncated = html_to_text(value)
    return text


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _crossref_year(issued: Any) -> str:
    parts = issued.get("date-parts") if isinstance(issued, dict) else None
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        first = parts[0][0]
        if isinstance(first, int) and not isinstance(first, bool):
            return str(first)
    return ""


def record_from_crossref_item(item: Any) -> dict | None:
    """Map one Crossref work to a source record; None for items without a DOI.

    ``url`` is the DOI URL, ``text`` the JATS-cleaned abstract or a brief
    metadata line, and ``sha256`` hashes the stored text.
    """
    if not isinstance(item, dict):
        return None
    doi = item.get("DOI")
    if not isinstance(doi, str) or not doi:
        return None
    url = f"https://doi.org/{doi}"
    title = _single_line(_strip_markup(item.get("title")))
    abstract = item.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        text = html_to_text(abstract)[0]
    else:
        parts = [part for part in (_single_line(_strip_markup(item.get("container-title"))), _crossref_year(item.get("issued"))) if part]
        text = ""
        if parts:
            text = "Crossref metadata: " + ". ".join(parts) + "."
    truncated = len(text) > MAX_TEXT_CHARS or _TRUNCATION_MARKER in text
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return {
        "id": source_id(url),
        "url": url,
        "title": title,
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retrieved_at": _now(),
        "truncated": truncated,
        "content_kind": "abstract" if isinstance(abstract, str) and abstract.strip() else "metadata",
    }


class SourceTools:
    """Fetch permitted public HTTPS text/HTML and search Crossref.

    Records are data, not instructions. fetch() follows at most three
    validated redirects, caps responses at 1 MiB, uses a 20 second timeout,
    and binds stored text to 32,000 characters. search() queries the fixed
    public Crossref works endpoint and returns at most five records.
    """

    def __init__(self):
        self.max_text_chars = MAX_TEXT_CHARS
        self.response_cap_bytes = RESPONSE_CAP_BYTES
        self.fetch_timeout_seconds = FETCH_TIMEOUT_SECONDS
        self.max_redirects = MAX_REDIRECTS

    def fetch(self, url: str) -> dict:
        """Retrieve a public HTTPS text/HTML document as a source record."""
        final_url, body, content_type = self._download(url)
        return build_source_record(final_url, body, content_type)

    def search(self, query: str, limit: int = SEARCH_RESULT_LIMIT) -> list[dict]:
        """Search Crossref and return source records with DOI URLs."""
        if not isinstance(query, str) or not query.strip():
            raise SourceError("A nonempty search query is required")
        if type(limit) is not int or not 1 <= limit <= SEARCH_RESULT_LIMIT:
            raise SourceError(f"limit must be an integer from 1 to {SEARCH_RESULT_LIMIT}")
        _url, body, _content_type = self._download(crossref_query_url(query.strip(), limit))
        try:
            payload = load_strict_json(body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise SourceError(f"Crossref response is not valid JSON: {exc}") from exc
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            raise SourceError("Crossref response has no items list")
        records = []
        for item in items[:limit]:
            record = record_from_crossref_item(item)
            if record is not None:
                records.append(record)
        return records

    def _download(self, url: str) -> tuple[str, bytes, str]:
        """GET the URL, following at most max_redirects re-vetted https redirects.

        Returns (final_url, body, content_type).
        """
        current = validate_https_url(url)
        for _hop in range(self.max_redirects + 1):
            outcome = self._request_once(current)
            if isinstance(outcome, str):
                current = outcome
                continue
            return outcome
        raise SourceError(f"More than {self.max_redirects} redirects were requested from {url}")

    def _request_once(self, url: str) -> tuple[str, bytes, str] | str:
        """Perform one GET against vetted addresses.

        Returns (url, body, content_type) on success or the next URL on a
        redirect. Connection errors try the next vetted address; protocol
        errors raise immediately.
        """
        parts = urlsplit(url)
        addresses = resolve_addresses(parts.hostname or "")
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        last_error: Exception | None = None
        for address in addresses:
            connection = _PinnedHTTPSConnection(address, parts.hostname or "", port=parts.port, timeout=self.fetch_timeout_seconds)
            try:
                connection.request("GET", path, headers=_FETCH_HEADERS)
                response = connection.getresponse()
                if response.status in REDIRECT_STATUSES:
                    return redirect_target(url, response.getheader("Location") or "")
                if response.status != 200:
                    raise SourceError(f"HTTP {response.status} from {url}")
                body = bounded_read(response, self.response_cap_bytes)
                return url, body, response.getheader("Content-Type") or ""
            except SourceError:
                raise
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
        raise SourceError(f"Could not connect to {parts.hostname} at {len(addresses)} vetted address(es): {last_error}")
