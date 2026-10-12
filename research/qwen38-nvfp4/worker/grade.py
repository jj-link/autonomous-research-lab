"""Study-specific official scoring; candidate execution is container-only.

LCB's reliability_guard is not a sandbox or a scoring-integrity boundary: the
candidate and private tests share an interpreter. Isolation protects the host,
not the scorer from adversarial introspection/tampering. No answer repair is
performed. The study's prompt preopens <think>, so a closing tag is required.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback

PER_TEST_SECONDS = 6
MAX_RESULT_BYTES = 1024 * 1024
LOG_BYTES = 16384
EXTRACTION_RULE = "qwen38-v1:after-last-think-close;last-python-or-untagged-fence;no-repair"
INTEGRITY_LIMIT = (
    "Official LCB executes candidate and private tests in the same interpreter; "
    "its reliability_guard is not a sandbox and does not establish adversarial "
    "scoring integrity. The enclosing offline container is the security boundary."
)


def extract_final_answer(text: str, *, thinking: bool = True) -> dict:
    """Keep final bytes unchanged, including whitespace relevant to IFEval.

    An implicit opening tag is allowed because the rendered prompt preopens it.
    Non-thinking mode is explicit, for callers outside this fixed study protocol.
    """
    if not isinstance(text, str):
        raise ValueError("generation.text must be a string")
    if "</think>" in text:
        final = text.rsplit("</think>", 1)[1]
        if "<think>" in final:
            return {"text": "", "status": "unfinished_reasoning", "rule": EXTRACTION_RULE}
    elif thinking or "<think>" in text:
        return {"text": "", "status": "unfinished_reasoning", "rule": EXTRACTION_RULE}
    else:
        final = text
    return {"text": final, "status": "ok" if final.strip() else "empty_final", "rule": EXTRACTION_RULE}


def extract_code(final: str) -> dict:
    """Select exactly the last Python/untagged block, or an unfenced answer.

    Fences are line-oriented Markdown backtick/tilde fences. A last eligible
    unclosed fence fails extraction rather than falling back to earlier code.
    Prose is never stripped to recover an otherwise invalid Python answer.
    ast.parse validates syntax only; it never executes candidate code.
    """
    opened = None
    selected = None
    saw_fence = False
    for line in final.splitlines(keepends=True):
        marker = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*", line)
        if opened is None:
            if marker:
                saw_fence = True
                fence, language = marker.groups()
                eligible = language.strip().lower() in ("", "python")
                opened = {"fence": fence, "eligible": eligible, "body": []}
                if eligible:
                    selected = {"text": "", "status": "unclosed_fence", "rule": EXTRACTION_RULE}
        elif marker and not marker.group(2).strip() and marker.group(1)[0] == opened["fence"][0] and len(marker.group(1)) >= len(opened["fence"]):
            if opened["eligible"]:
                selected = {"text": "".join(opened["body"]), "status": "ok", "rule": EXTRACTION_RULE}
            opened = None
        else:
            opened["body"].append(line)
    if saw_fence:
        if selected is None:
            return {"text": "", "status": "no_python_fence", "rule": EXTRACTION_RULE}
        if selected["status"] != "ok":
            return selected
        code = selected["text"]
        source = "fenced"
    else:
        code = final
        source = "unfenced"
    result = {"text": code, "status": "ok", "source": source, "rule": EXTRACTION_RULE}
    if not code.strip():
        result["status"] = "empty_code"
        return result
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        result.update(status="invalid_python", error=str(exc))
        return result
    if not any(not isinstance(node, ast.Expr) or isinstance(node.value, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom)) for node in tree.body):
        result["status"] = "no_executable_code"
    return result


def assess_response(generation: dict, final: dict) -> dict:
    """Completion events are orthogonal to observed task correctness."""
    text = generation.get("text", "")
    finish = generation.get("finish_reason")
    error = generation.get("error")
    empty = not isinstance(text, str) or not text.strip()
    truncated = finish in ("length", "max_tokens")
    final_empty = not final["text"].strip()
    normal = error is None and not empty and not truncated and finish == "stop" and final["status"] == "ok"
    if error is not None:
        category = "generation_error"
    elif empty:
        category = "empty"
    elif truncated:
        category = "truncated"
    elif final["status"] != "ok":
        category = final["status"]
    elif finish != "stop":
        category = "abnormal_finish"
    else:
        category = "completed"
    return {"category": category, "normal_completion": normal, "empty": empty,
            "empty_final": final_empty, "truncated": truncated, "generation_error": error is not None,
            "finish_reason": finish, "error": error, "token_count": len(generation.get("token_ids") or [])}


def lcb_test_count(sample: dict) -> int:
    data = json.loads(sample["input_output"])
    inputs, outputs = data["inputs"], data["outputs"]
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        raise ValueError("LCB requires nonempty equal-length input/output lists")
    if not all(isinstance(value, str) for value in inputs + outputs):
        raise ValueError("LCB test inputs and outputs must be strings")
    if data.get("fn_name") is not None and not isinstance(data["fn_name"], str):
        raise ValueError("LCB fn_name must be null or a string")
    return len(inputs)


def lcb_wall_seconds(test_count: int) -> int:
    if type(test_count) is not int or test_count <= 0:
        raise ValueError("test_count must be positive")
    # Same allowance as the pinned official check_correctness wrapper.
    return (PER_TEST_SECONDS + 1) * test_count + 5


def classify_lcb_results(results: list, expected_tests: int) -> dict:
    """Use the official >0 predicate, never Python truthiness of error codes.

    The official harness short-circuits on first failure, so failed prefixes are
    valid; an all-positive short prefix is not evidence that all tests passed.
    """
    if not isinstance(results, list) or not results or type(expected_tests) is not int or expected_tests <= 0:
        raise ValueError("Missing official test results or empty test inventory")
    if len(results) > expected_tests:
        raise ValueError("Scorer returned more results than tests")
    if any(type(value) not in (bool, int, float) or (type(value) is float and not math.isfinite(value)) for value in results):
        raise ValueError("Official test results must be finite numeric values")
    passed = [value > 0 for value in results]
    if all(passed) and len(results) != expected_tests:
        raise ValueError("All-positive result omitted tests")
    category = "passed" if all(passed) else "wrong_answer"
    for code, label in ((-5, "test_runner_error"), (-1, "global_timeout"), (-4, "runtime_error"), (-3, "test_timeout")):
        if code in results:
            category = label
            break
    return {"correctness": all(passed), "category": category, "per_test": results,
            "test_passed": passed, "expected_tests": expected_tests,
            "tested_count": len(results), "unrun_count": expected_tests - len(results)}


def _failure(kind: str, message: str, **evidence) -> dict:
    return {"status": "failed", "correctness": None,
            "error": {"kind": kind, "message": message}, **evidence}


def _require_container() -> None:
    if not sys.platform.startswith("linux") or os.environ.get("ARL_ISOLATED_GRADING") != "1":
        raise RuntimeError("Coding execution requires the parent-provided isolated Linux grading container (ARL_ISOLATED_GRADING=1)")


def _log_evidence(stream) -> dict:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - LOG_BYTES))
    return {"bytes": size, "tail": stream.read().decode("utf-8", errors="replace"), "truncated": size > LOG_BYTES}


def run_lcb(sample: dict, code: str) -> dict:
    """Run the unmodified official API in a disposable child, never the parent."""
    _require_container()
    count = lcb_test_count(sample)
    allowance = lcb_wall_seconds(count)
    read_fd, write_fd = os.pipe()
    process = None
    payload = bytearray()
    timed_out = False
    oversized = False
    try:
        with tempfile.TemporaryFile(dir="/dev/shm") as out, tempfile.TemporaryFile(dir="/dev/shm") as err:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--lcb-child", str(write_fd)],
                stdin=subprocess.PIPE, stdout=out, stderr=err, pass_fds=(write_fd,), start_new_session=True,
            )
            os.close(write_fd)
            write_fd = None
            deadline = time.monotonic() + allowance
            request = memoryview(json.dumps({"sample": sample, "code": code}).encode("utf-8"))
            sent = 0
            os.set_blocking(process.stdin.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ, "result")
                selector.register(process.stdin.fileno(), selectors.EVENT_WRITE, "request")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        if key.data == "request":
                            try:
                                sent += os.write(key.fd, request[sent:sent + 65536])
                            except BrokenPipeError:
                                sent = len(request)
                            if sent == len(request):
                                selector.unregister(key.fd)
                                process.stdin.close()
                            continue
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fd)
                        else:
                            payload.extend(chunk)
                            if len(payload) > MAX_RESULT_BYTES:
                                oversized = True
                                break
                    if oversized:
                        break
                if not timed_out and not oversized:
                    try:
                        process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        timed_out = True
            # Kill the entire group, including any descendant retaining the pipe.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            evidence = {"exit_code": process.returncode, "stdout": _log_evidence(out), "stderr": _log_evidence(err),
                        "per_test_seconds": PER_TEST_SECONDS, "wall_seconds": allowance, "expected_tests": count,
                        "integrity_limit": INTEGRITY_LIMIT}
            if timed_out:
                return _failure("scorer_timeout", "Official scorer child exceeded whole-call allowance", **evidence)
            if oversized:
                return _failure("scorer_protocol", "Child result exceeded JSON result byte limit", **evidence)
            if process.returncode != 0:
                return _failure("scorer_exit", "Official scorer child exited abnormally", **evidence)
            try:
                response = json.loads(payload)
                if response["status"] == "failed":
                    return {**response, **evidence}
                classified = classify_lcb_results(response["per_test"], count)
                return {"status": "completed", **classified, "metadata": response["metadata"], **evidence}
            except (ValueError, TypeError, KeyError) as exc:
                return _failure("scorer_protocol", str(exc), response_tail=payload[-LOG_BYTES:].decode("utf-8", errors="replace"), **evidence)
    finally:
        if process is not None and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def _lcb_child(result_fd: int) -> None:
    _require_container()
    # Retain these before the official guard mutates process state. The pipe
    # carries JSON, not pickle, so corrupt child data cannot execute in parent.
    write_bytes, encode_json = os.write, json.dumps
    try:
        request = json.load(sys.stdin)
        from lcb_runner.evaluation.testing_util import run_test
        results, metadata = run_test(request["sample"], test=request["code"], debug=False, timeout=PER_TEST_SECONDS)
        # Official results may contain numpy scalar bools/integers.
        results = [value.item() if hasattr(value, "item") else value for value in results]
        response = {"status": "completed", "per_test": results, "metadata": metadata}
    except BaseException as exc:
        response = _failure("scorer_exception", repr(exc), traceback=traceback.format_exc())
    raw = encode_json(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
    while raw:
        raw = raw[write_bytes(result_fd, raw):]


def run_ifeval(grading: dict, final: str) -> dict:
    from instruction_following_eval import evaluation_lib
    import random
    from langdetect import DetectorFactory
    random.seed(0)
    DetectorFactory.seed = 0
    ids, kwargs = grading["instruction_id_list"], grading["kwargs"]
    if not isinstance(ids, list) or not isinstance(kwargs, list) or not ids or len(ids) != len(kwargs):
        raise ValueError("IFEval requires nonempty matching instructions and kwargs")
    if type(grading["key"]) is not int or not isinstance(grading["prompt"], str) or not all(isinstance(value, str) for value in ids) or not all(isinstance(value, dict) for value in kwargs):
        raise ValueError("Malformed official IFEval input")
    example = evaluation_lib.InputExample(key=grading["key"], instruction_id_list=ids, prompt=grading["prompt"], kwargs=kwargs)
    response = {"status": "completed", "correctness": None, "instruction_ids": ids, "errors": {}, "evaluator_seed": 0}
    for mode, scorer in (("strict", evaluation_lib.test_instruction_following_strict), ("loose", evaluation_lib.test_instruction_following_loose)):
        try:
            scored = scorer(example, {example.prompt: final})
            flags = scored.follow_instruction_list
            if len(flags) != len(ids) or any(type(value) is not bool for value in flags) or scored.follow_all_instructions != all(flags):
                raise ValueError("Inconsistent official IFEval result")
            response[mode] = {"prompt_passed": scored.follow_all_instructions, "per_instruction": flags,
                              "instruction_fraction": sum(flags) / len(flags)}
            if mode == "strict":
                response["correctness"] = scored.follow_all_instructions
        except Exception as exc:
            response["errors"][mode] = {"kind": "scorer_exception", "message": repr(exc), "traceback": traceback.format_exc()}
    if response["errors"]:
        response["status"] = "failed"
    return response


def grade_job(job: dict) -> dict:
    """Full grading report; a null correctness denotes unavailable evidence."""
    family, identity = job["family"], job["id"]
    if family not in ("coding", "ifeval"):
        raise ValueError("Unknown grading family")
    generation = job["generation"]
    final = extract_final_answer(generation.get("text", ""), thinking=generation.get("thinking_enabled", True))
    completion = assess_response(generation, final)
    extraction = extract_code(final["text"]) if family == "coding" and final["status"] == "ok" else final
    scorer = {"status": "not_run", "correctness": None, "reason": "unavailable_final_answer"}
    if job.get("grading_error") is not None:
        scorer = job["grading_error"]
    elif not completion["generation_error"] and extraction["status"] == "ok":
        try:
            scorer = run_lcb(job["grading"], extraction["text"]) if family == "coding" else run_ifeval(job["grading"], extraction["text"])
        except Exception as exc:
            scorer = _failure("scorer_exception", repr(exc), traceback=traceback.format_exc())
    correct = scorer["correctness"]
    metrics = {"correct": int(correct is True), "normal_completion": int(completion["normal_completion"]),
               "generation_error": int(completion["generation_error"]), "empty": int(completion["empty"]),
               "empty_final": int(completion["empty_final"]), "truncated": int(completion["truncated"]),
               "extraction_failure": int(extraction["status"] != "ok"), "scorer_failure": int(scorer["status"] == "failed")}
    if family == "ifeval":
        for mode in ("strict", "loose"):
            outcome = scorer.get(mode, {})
            metrics[f"ifeval_{mode}_prompt"] = int(outcome.get("prompt_passed", False))
            metrics[f"ifeval_{mode}_instruction"] = outcome.get("instruction_fraction", 0.0)
    return {"metrics": metrics, "details": {"schema_version": 1, "id": identity, "family": family,
            "completion": completion, "extraction": extraction, "correctness": correct, "scorer": scorer}}


def write_reports(report: dict, directory: Path) -> None:
    """Keep large answer/evidence bodies outside the worker's 1-MiB summary."""
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    (directory / "grading.json").write_bytes(encoded)
    details = report["details"]
    scorer = details["scorer"]
    summary_scorer = {key: scorer[key] for key in (
        "status", "correctness", "category", "exit_code", "per_test_seconds",
        "wall_seconds", "expected_tests", "tested_count", "unrun_count", "evaluator_seed"
    ) if key in scorer}
    if "error" in scorer:
        summary_scorer["error"] = {"kind": scorer["error"]["kind"]}
    summary = {"metrics": report["metrics"], "details": {
        "schema_version": 1, "id": details["id"], "family": details["family"],
        "correctness": details["correctness"],
        "completion": {key: details["completion"][key] for key in (
            "category", "normal_completion", "empty", "empty_final", "truncated",
            "generation_error", "finish_reason", "token_count"
        )},
        "extraction": {key: details["extraction"][key] for key in ("status", "source", "rule") if key in details["extraction"]},
        "scorer": summary_scorer,
        "grading_artifact": {"path": "grading.json", "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()},
    }}
    (directory / "results.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def validate_grading_filename(filename: str) -> tuple[str, ...]:
    if not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename:
        raise ValueError("grading_file must be a relative POSIX filename")
    path = PurePosixPath(filename)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", filename) or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ValueError("grading_file must stay under /inputs/grading")
    return path.parts


def load_grading_file(filename: str) -> dict:
    """Open beneath the approved Linux mount without following any symlink.

    No JSON byte cap: private test inventories can exceed the manifest limit.
    The parent owns container memory limits and dataset admission.
    """
    parts = validate_grading_filename(filename)
    root_fd = os.open("/inputs/grading", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = root_fd
    try:
        for part in parts[:-1]:
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            if fd != root_fd:
                os.close(fd)
            fd = child_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(file_fd, "r", encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError("grading file must contain an object")
        return data
    finally:
        if fd != root_fd:
            os.close(fd)
        os.close(root_fd)


def main() -> None:
    with open("/experiment/settings.json", encoding="utf-8") as source:
        job = json.load(source)
    try:
        job["grading"] = load_grading_file(job["grading_file"])
    except Exception as exc:
        job["grading_error"] = _failure("grading_job_error", repr(exc), traceback=traceback.format_exc())
    write_reports(grade_job(job), Path("/workspace"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--lcb-child":
        _lcb_child(int(sys.argv[2]))
    else:
        main()
