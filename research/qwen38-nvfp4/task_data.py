"""Pure local task freezing for the pinned Qwen3.8 compatibility study.

Selection uses no model outputs. Every supplied row is validated before selecting
8 compatibility and 128 primary tasks per family. Only public projections may be
mounted into inference workers; ``private`` contains the official scorer inputs.

LCB decoding/sample layout follows code_generation.py at
28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24. Its compressed pickle is inspected as a
primitive string/bytes envelope, never unpickled or allowed to construct objects.
The Python prompt is study-specific, not an exact copy of LCB's model templates.
IFEval prompts are preserved byte-for-byte as Python strings, without wrapping.
"""

import base64
from bisect import insort
import binascii
import hashlib
import json
import pickletools
import zlib
from collections.abc import Mapping


NAMESPACE = "arl-qwen38-nvfp4-v1"
COMPATIBILITY_COUNT = 8
PRIMARY_COUNT = 128


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads(text, label):
    if not isinstance(text, (str, bytes)):
        raise ValueError(f"{label} must be JSON text")
    try:
        return json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Malformed {label}") from exc


def _primitive_pickle(payload):
    """Read only a pickle containing one literal text value and its memo write.

    GLOBAL, REDUCE, constructors, persistent IDs, and container opcodes are not
    interpreted. pickletools parses opcode arguments but executes no pickle code.
    This accepts primitive str/bytes pickles from protocols 0 through 5.
    """
    literal_ops = {
        "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
        "STRING", "BINSTRING", "SHORT_BINSTRING",
        "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
    }
    value = None
    memo_written = False
    stopped = False
    try:
        for opcode, argument, position in pickletools.genops(payload):
            name = opcode.name
            if name == "PROTO" and position == 0 and argument <= 5:
                continue
            if name == "FRAME":
                if position + 9 + argument > len(payload):
                    raise ValueError("Truncated pickle frame")
                continue
            if name in literal_ops and value is None:
                value = argument
                continue
            if name in {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT"}:
                if value is None or memo_written:
                    raise ValueError("Invalid primitive pickle memo")
                memo_written = True
                continue
            if name == "STOP" and isinstance(value, (str, bytes)):
                if position + 1 != len(payload):
                    raise ValueError("Trailing pickle bytes")
                stopped = True
                continue
            raise ValueError(f"Unsupported private-test pickle opcode: {name}")
    except (ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("Private tests require a primitive text pickle") from exc
    if not stopped:
        raise ValueError("Missing primitive pickle value")
    return value


def _private_tests(text):
    if not isinstance(text, str):
        raise ValueError("private_test_cases must be text")
    try:
        return _json_loads(text, "private_test_cases")
    except ValueError:
        # The pinned loader falls back to base64 -> zlib -> pickle -> JSON.
        # Unlike that loader, no pickle opcode is ever executed here.
        try:
            compressed = base64.b64decode(text.encode("ascii"), validate=True)
            decoder = zlib.decompressobj()
            payload = decoder.decompress(compressed) + decoder.flush()
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("Invalid compressed private-test envelope")
            return _json_loads(_primitive_pickle(payload), "private_test_cases")
        except (ValueError, UnicodeError, binascii.Error, zlib.error) as exc:
            raise ValueError("Malformed private_test_cases") from exc


def _text(value, label, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be {'possibly empty ' if allow_empty else 'nonempty '}text")
    return value


def _tests(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    for test in value:
        if not isinstance(test, dict) or set(test) != {"input", "output", "testtype"}:
            raise ValueError(f"Malformed {label} test")
        _text(test["input"], f"{label} input", allow_empty=True)
        _text(test["output"], f"{label} output", allow_empty=True)
        if test["testtype"] not in ("stdin", "functional"):
            raise ValueError(f"Unsupported {label} testtype")
    return value


def render_coding_prompt(question_content, starter_code, public_test_cases):
    """Render Python instructions using only question, starter and public tests.

    Functional starter code is preserved; empty/null starter code means a stdin
    and stdout solution, as in LCB's template. Examples are JSON-quoted strings.
    The grading adapter extracts the last Python/untagged fenced answer block.
    """
    _text(question_content, "question_content")
    if starter_code is not None:
        _text(starter_code, "starter_code", allow_empty=True)
    _tests(public_test_cases, "public_test_cases")
    parts = [
        "Solve the following programming problem in Python 3.",
        "### Question\n" + question_content,
    ]
    if starter_code:
        parts.append(
            "### Solution interface\nUse the following starter code and preserve its "
            "function/class interface.\n```python\n" + starter_code + "\n```"
        )
    else:
        parts.append(
            "### Solution interface\nRead input from standard input and write the "
            "answer to standard output. The program must run the solution when "
            "executed; do not hard-code the examples."
        )
    if public_test_cases:
        examples = ["### Public examples\nInputs and expected outputs are JSON-quoted strings."]
        for index, test in enumerate(public_test_cases, 1):
            examples.append(
                f"Example {index}:\nInput: {json.dumps(test['input'], ensure_ascii=False)}"
                f"\nExpected output: {json.dumps(test['output'], ensure_ascii=False)}"
            )
        parts.append("\n\n".join(examples))
    parts.append(
        "### Answer format\nReturn the complete solution in one fenced code block "
        "tagged python. Do not include test runs or explanatory prose in that block."
    )
    return "\n\n".join(parts)


def _project(family, dataset_id, prompt, grading):
    task_id = f"{family}/{dataset_id}"
    messages = [{"role": "user", "content": prompt}]
    digest = hashlib.sha256(json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    identity = {"id": task_id, "family": family, "dataset_id": dataset_id}
    return (
        {**identity, "messages": messages, "prompt_sha256": digest},
        {**identity, "grading": grading},
    )


def _coding(row):
    platform = row["platform"]
    if platform not in ("leetcode", "codeforces", "atcoder"):
        raise ValueError("Unsupported coding platform")
    question_id = _text(row["question_id"], "question_id")
    if "/" in question_id:
        raise ValueError("question_id must not contain a slash")
    public = _tests(_json_loads(row["public_test_cases"], "public_test_cases"), "public_test_cases")
    private = _tests(_private_tests(row["private_test_cases"]), "private_test_cases")
    if not public and not private:
        raise ValueError("Coding task has no evaluation tests")
    metadata = _json_loads(row["metadata"], "metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    function_name = metadata.get("func_name")
    if function_name is not None:
        _text(function_name, "func_name")
    tests = public + private
    # Match get_evaluation_sample(), including default json.dumps formatting.
    grading = {"input_output": json.dumps({
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
        "fn_name": function_name,
    })}
    return _project(
        "coding", f"{platform}/{question_id}",
        render_coding_prompt(row["question_content"], row["starter_code"], public), grading,
    )


def _validate_json_value(value):
    if value is None or type(value) in (str, int, float, bool):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError("Grading arguments must contain only JSON values and string keys")


def _ifeval(row):
    key = row["key"]
    if type(key) is not int:
        raise ValueError("IFEval key must be an integer")
    prompt = _text(row["prompt"], "IFEval prompt")
    instructions = row["instruction_id_list"]
    kwargs = row["kwargs"]
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("IFEval instruction_id_list must be nonempty")
    for instruction in instructions:
        _text(instruction, "IFEval instruction ID")
    if not isinstance(kwargs, list) or len(kwargs) != len(instructions):
        raise ValueError("IFEval kwargs must match instruction_id_list length")
    if not all(isinstance(item, dict) for item in kwargs):
        raise ValueError("IFEval kwargs entries must be objects")
    try:
        _validate_json_value(kwargs)
        # Retain original evaluator fields, while detaching nested caller data.
        grading = json.loads(json.dumps({
            "key": key, "instruction_id_list": instructions,
            "prompt": prompt, "kwargs": kwargs,
        }, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("IFEval grading fields must be JSON data") from exc
    return _project("ifeval", str(key), prompt, grading)


def prepare_tasks(coding_rows, ifeval_rows):
    """Freeze exact 8+128 per-family splits, rejecting any bad or duplicate row.

    Both inputs are iterables of raw dataset mappings. No rows are silently
    filtered: compatibility rows and all unselected rows are excluded from the
    primary split solely by the fixed outcome-independent ranking.
    Rows are consumed incrementally; only the best-ranked 136 projections per
    family are retained, rather than all expanded private test inventories.
    Selection lists contain full task IDs, not abbreviated dataset IDs.
    """
    public = {"compatibility": [], "primary": []}
    private = {"compatibility": [], "primary": []}
    selection = {"rule": {
        "namespace": NAMESPACE,
        "rank": "sha256_utf8_then_dataset_id",
        "compatibility_count": COMPATIBILITY_COUNT,
        "primary_count": PRIMARY_COUNT,
    }}
    for family, rows, projector in (
        ("coding", coding_rows, _coding), ("ifeval", ifeval_rows, _ifeval)
    ):
        seen = set()
        ranked = []
        count = COMPATIBILITY_COUNT + PRIMARY_COUNT
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{family} rows must be mappings")
            try:
                visible, hidden = projector(row)
            except KeyError as exc:
                raise ValueError(f"Missing {family} field: {exc.args[0]}") from exc
            dataset_id = visible["dataset_id"]
            if dataset_id in seen:
                raise ValueError(f"Duplicate {family} dataset ID: {dataset_id}")
            seen.add(dataset_id)
            rank = hashlib.sha256(f"{NAMESPACE}/{family}/{dataset_id}".encode("utf-8")).hexdigest()
            insort(ranked, (rank, dataset_id, visible, hidden))
            if len(ranked) > count:
                ranked.pop()
        if len(seen) < count:
            raise ValueError(f"Insufficient {family} pool: need {count}, got {len(seen)}")
        selection[family] = {}
        for split, entries in (
            ("compatibility", ranked[:COMPATIBILITY_COUNT]),
            ("primary", ranked[COMPATIBILITY_COUNT:count]),
        ):
            selection[family][split] = []
            for _, dataset_id, visible, hidden in entries:
                public[split].append(visible)
                private[split].append(hidden)
                selection[family][split].append(visible["id"])
    return {"public": public, "private": private, "selection": selection}
