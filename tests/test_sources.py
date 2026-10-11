"""Pure parsing and security boundaries of the source tools, plus SSRF guards exercised without network.

fetch() refuses loopback/private/redirect targets by design, so no end-to-end
retrieval test is possible offline; instead the exact decision functions fetch
uses (URL validation, address vetting, redirect resolution, pinning, text
normalization, record building, Crossref mapping) are called directly. All
literal-IP and localhost DNS cases resolve without touching the network.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import re
from datetime import datetime
from unittest import TestCase

from autonomous_research_lab.sources import (
    MAX_TEXT_CHARS,
    RESPONSE_CAP_BYTES,
    SourceError,
    SourceTools,
    _PinnedHTTPSConnection,
    _TRUNCATION_MARKER,
    bounded_read,
    crossref_query_url,
    html_to_text,
    is_permitted_address,
    load_strict_json,
    record_from_crossref_item,
    redirect_target,
    resolve_addresses,
    validate_https_url,
    urlsplit,
)

PRIVATE_HOSTS = [
    "127.0.0.1",       # loopback
    "10.0.0.1",        # private
    "172.16.0.1",      # private
    "192.168.1.1",     # private
    "169.254.169.254", # link-local (cloud metadata)
    "0.0.0.0",         # unspecified
    "224.0.0.1",       # multicast (is_global can be True for multicast)
    "100.64.0.1",      # carrier-grade NAT
    "192.0.2.1",       # TEST-NET-1
    "240.0.0.1",       # reserved
    "::1",             # IPv6 loopback
    "fe80::1",         # IPv6 link-local
    "::ffff:10.0.0.1", # IPv4-mapped IPv6 must obey IPv4 rules
]


class ValidateHttpsUrlTests(TestCase):
    def test_accepts_plain_https_and_strips_fragments(self):
        self.assertEqual(validate_https_url("https://example.com/a/b?x=1"), "https://example.com/a/b?x=1")
        self.assertEqual(validate_https_url("https://example.com/page#section"), "https://example.com/page")
        self.assertEqual(validate_https_url("  https://example.com/  "), "https://example.com/")

    def test_rejects_non_https_schemes(self):
        for url in ("http://example.com/", "ftp://example.com/file", "file:///etc/passwd", "https-missing.example.com"):
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    validate_https_url(url)

    def test_rejects_embedded_credentials(self):
        for url in ("https://user:pass@example.com/", "https://token@example.com/", "https://:pass@example.com/"):
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    validate_https_url(url)

    def test_rejects_missing_host_and_non_strings(self):
        for url in ("https:///path", "", "   ", 42, None):
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    validate_https_url(url)

    def test_malformed_authorities_fail_before_network_access(self):
        for url in ("https://example.com:bad/", "https://example.com:65536/", "https://[broken/", "https://example.com/a\nb"):
            with self.subTest(url=url), self.assertRaises(SourceError):
                validate_https_url(url)


class AddressVettingTests(TestCase):
    def test_every_private_shape_is_refused(self):
        for host in PRIVATE_HOSTS:
            with self.subTest(host=host):
                with self.assertRaises(SourceError):
                    resolve_addresses(host)

    def test_dns_private_case_localhost_is_refused(self):
        with self.assertRaises(SourceError):
            resolve_addresses("localhost")

    def test_public_literals_are_permitted_without_network(self):
        self.assertEqual(resolve_addresses("8.8.8.8"), ["8.8.8.8"])
        self.assertEqual(resolve_addresses("2606:4700:4700::1111"), ["2606:4700:4700::1111"])

    def test_is_permitted_address_classification(self):
        self.assertTrue(is_permitted_address(ipaddress.ip_address("8.8.8.8")))
        self.assertFalse(is_permitted_address(ipaddress.ip_address("224.0.0.1")))
        self.assertFalse(is_permitted_address(ipaddress.ip_address("::ffff:169.254.169.254")))


class RedirectTests(TestCase):
    def test_relative_and_scheme_relative_locations_resolve(self):
        self.assertEqual(redirect_target("https://example.com/a/b", "/next?x=1"), "https://example.com/next?x=1")
        self.assertEqual(redirect_target("https://example.com/a", "//other.example.com/p"), "https://other.example.com/p")
        self.assertEqual(redirect_target("https://example.com/a", "https://other.example.com/p"), "https://other.example.com/p")

    def test_downgrade_and_credential_locations_are_rejected(self):
        for location in ("http://other.example.com/p", "https://user:pass@other.example.com/", "", "   ", None):
            with self.subTest(location=location):
                with self.assertRaises(SourceError):
                    redirect_target("https://example.com/a", location)

    def test_redirect_to_private_host_is_refused_by_vetting(self):
        # fetch validates the redirect target, then vets its host before connecting.
        target = redirect_target("https://example.com/a", "https://169.254.169.254/latest/meta-data")
        with self.assertRaises(SourceError):
            resolve_addresses(urlsplit(target).hostname)

    def test_fetch_refuses_https_private_targets_before_connecting(self):
        for url in ("https://127.0.0.1/secret", "https://10.9.8.7/internal", "https://localhost/x"):
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    SourceTools().fetch(url)

    def test_fetch_refuses_plain_http_and_credentials_upfront(self):
        for url in ("http://example.com/", "https://user:pass@example.com/"):
            with self.subTest(url=url):
                with self.assertRaises(SourceError):
                    SourceTools().fetch(url)




class HtmlNormalizationTests(TestCase):
    def test_strips_script_and_style_and_extracts_title(self):
        page = (
            "<html><head><style>p { color: red }</style><title>Hello World</title>"
            "<script>var secrets = 'x';</script></head>"
            "<body><p>Visible text</p><script>alert(1)</script></body></html>"
        )
        text, title, truncated = html_to_text(page)
        self.assertEqual(text, "Visible text")
        self.assertEqual(title, "Hello World")
        self.assertFalse(truncated)
        self.assertNotIn("secrets", text)
        self.assertNotIn("alert", text)

    def test_decodes_entities_and_separates_blocks(self):
        text, _title, _truncated = html_to_text("<p>Fish &amp; chips &lt;3</p><p>Second</p><div>tail</div>")
        self.assertEqual(text, "Fish & chips <3\nSecond\ntail")

    def test_jats_marked_up_abstracts_are_cleaned(self):
        text, _title, _truncated = html_to_text("<jats:p>One finding.</jats:p><jats:title>Intro</jats:title><jats:italic>keep</jats:italic>")
        self.assertEqual(text, "One finding.\nIntro\nkeep")

    def test_missing_title_is_empty(self):
        self.assertEqual(html_to_text("<p>Only body</p>"), ("Only body", "", False))

    def test_large_text_is_bounded_and_marked(self):
        text, _title, truncated = html_to_text("<p>" + "x" * (MAX_TEXT_CHARS + 5000) + "</p>")
        self.assertTrue(truncated)
        self.assertLessEqual(len(text), MAX_TEXT_CHARS)
        self.assertTrue(text.endswith(_TRUNCATION_MARKER))
        self.assertIn("[Truncated", text)

    def test_normal_pages_are_not_marked_truncated(self):
        text, _title, truncated = html_to_text("<p>" + "y" * 1000 + "</p>")
        self.assertFalse(truncated)
        self.assertNotIn("[Truncated", text)
        self.assertEqual(len(text), 1000)


class BoundedReadTests(TestCase):
    def test_reads_up_to_cap(self):
        self.assertEqual(bounded_read(io.BytesIO(b"x" * RESPONSE_CAP_BYTES), RESPONSE_CAP_BYTES), b"x" * RESPONSE_CAP_BYTES)

    def test_rejects_beyond_cap(self):
        with self.assertRaises(SourceError):
            bounded_read(io.BytesIO(b"x" * (RESPONSE_CAP_BYTES + 1)), RESPONSE_CAP_BYTES)


class SourceRecordTests(TestCase):
    def test_record_shape_and_hashes(self):
        body = b"<html><head><title>Doc</title></head><body><p>Hi</p></body></html>"
        from autonomous_research_lab.sources import build_source_record
        record = build_source_record("https://example.com/doc", body, "text/html; charset=utf-8")
        self.assertEqual(record["title"], "Doc")
        self.assertEqual(record["text"], "Hi")
        self.assertEqual(record["sha256"], hashlib.sha256(body).hexdigest())
        self.assertIsInstance(datetime.fromisoformat(record["retrieved_at"]), datetime)

    def test_ids_are_stable_per_url(self):
        from autonomous_research_lab.sources import build_source_record
        first = build_source_record("https://example.com/a", b"one", "text/plain")
        second = build_source_record("https://example.com/a", b"two", "text/plain")
        third = build_source_record("https://example.com/b", b"one", "text/plain")
        self.assertEqual(first["id"], second["id"])
        self.assertNotEqual(first["id"], third["id"])

    def test_plain_text_is_bounded_and_marked(self):
        from autonomous_research_lab.sources import build_source_record
        record = build_source_record("https://example.com/big", b"z" * (MAX_TEXT_CHARS + 10), "text/plain")
        self.assertLessEqual(len(record["text"]), MAX_TEXT_CHARS)
        self.assertIn("[Truncated", record["text"])
        record = build_source_record("https://example.com/small", "héllo".encode("utf-8"), "text/plain")
        self.assertEqual(record["text"], "héllo")

    def test_explicit_and_unknown_charsets(self):
        from autonomous_research_lab.sources import build_source_record
        self.assertEqual(build_source_record("https://e.com/", "héllo".encode("latin-1"), "text/plain; charset=latin-1")["text"], "héllo")
        self.assertEqual(build_source_record("https://e.com/", "héllo".encode("utf-8"), "text/plain; charset=nonsense")["text"], "héllo")

    def test_unsupported_media_is_rejected(self):
        from autonomous_research_lab.sources import build_source_record
        for content_type in ("", "image/png", "application/pdf", "application/octet-stream", "application/json", "application/javascript"):
            with self.subTest(content_type=content_type):
                with self.assertRaises(SourceError):
                    build_source_record("https://e.com/", b"\x89PNG", content_type)


class CrossrefTests(TestCase):

    def test_search_validates_query_and_limit_before_any_request(self):
        for query, limit in (("", 5), ("   ", 5), ("topic", 0), ("topic", 6), ("topic", "3"), ("topic", 2.5), (None, 5)):
            with self.subTest(query=query, limit=limit):
                with self.assertRaises(SourceError):
                    SourceTools().search(query, limit)

    def test_item_maps_to_record_with_doi_url_and_cleaned_abstract(self):
        item = {
            "DOI": "10.9999/test.1",
            "title": ["A  Study of <italic>Things</italic>"],
            "abstract": "<jats:p>The findings are clear.</jats:p>",
            "container-title": ["Journal of Tests"],
            "issued": {"date-parts": [[2024, 5, 1]]},
        }
        record = record_from_crossref_item(item)
        self.assertEqual(record["title"], "A Study of Things")
        self.assertEqual(record["text"], "The findings are clear.")
        self.assertEqual(record["sha256"], hashlib.sha256(record["text"].encode("utf-8")).hexdigest())

    def test_items_without_doi_are_skipped(self):
        self.assertIsNone(record_from_crossref_item({"title": ["No DOI"]}))
        self.assertIsNone(record_from_crossref_item("not a dict"))


    def test_strict_json_rejects_non_finite_constants(self):
        self.assertEqual(load_strict_json("{\"a\": 1}"), {"a": 1})
        for text in ("{\"a\": NaN}", "{\"a\": Infinity}", "{\"a\": -Infinity}"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    load_strict_json(text)

    def test_composed_metadata_respects_the_stored_text_limit(self):
        record = record_from_crossref_item({"DOI": "10.1/long", "container-title": ["x" * MAX_TEXT_CHARS], "issued": {"date-parts": [[2024]]}})
        self.assertLessEqual(len(record["text"]), MAX_TEXT_CHARS)
        self.assertTrue(record["truncated"])
        self.assertEqual(record["content_kind"], "metadata")



if __name__ == "__main__":
    unittest.main()
