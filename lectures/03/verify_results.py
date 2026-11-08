"""保存済みDay3ログを検算。APIを呼ばない。"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = BASE / "results" / "20261108T155300+0900"
CASES = ["01_plain", "02_agent", "03_quiz", "04_before", "05_after"]
records = {name: json.loads((RESULTS / f"{name}.json").read_text()) for name in CASES}


def parts(record, kind):
    return [p for m in record.get("messages", []) for p in m["parts"] if p["part_kind"] == kind]


expected_calls = {
    "02_agent": [("search_glossary", {"query": "lecNN-start タグ"})],
    "03_quiz": [("make_quiz", {"lecture": 2})],
    "04_before": [("search_glossary", {"query": "LoRA"}), ("search_glossary", {"query": "QLoRA"})],
    "05_after": [("search_glossary", {"query": "LoRA"}), ("search_glossary", {"query": "QLoRA"})],
}
for name, record in records.items():
    assert record["status"] == "success" and record["output"]
    assert record["model"] == "gemini-3.5-flash-lite"
    for filename, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    trace = (RESULTS / f"{name}.txt").read_text()
    assert record["output"] in trace
    if name != "01_plain":
        calls = parts(record, "tool-call")
        assert [(p["tool_name"], p["args"]) for p in calls] == expected_calls[name]
        returns = parts(record, "tool-return")
        assert {p["tool_call_id"] for p in calls} == {p["tool_call_id"] for p in returns}
        assert record["requests"] == sum(m["kind"] == "response" for m in record["messages"]) == 2
        assert trace.count("[tool call]") == len(calls)
    else:
        assert record["requests"] == 1
before, after = records["04_before"], records["05_after"]
removed = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
assert before["instructions"].count(removed) == 1
assert before["instructions"].replace(removed, "") == after["instructions"]
assert before["question"] == after["question"] == "LoRAとQLoRAの違いは？"
assert records["01_plain"]["question"] == records["02_agent"]["question"]
print("PASS: 5 cases; tool calls, requests, paired questions, single instruction change, trace and source hashes")
