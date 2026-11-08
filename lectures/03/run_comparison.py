"""Day3の実測を保存する。無料枠を確認後、--confirm-free-tier --traceで実行。"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from app import tools
from app.agent import INSTRUCTIONS, build_model, print_trace
from app.ask import ask
from app.config import MODEL

REMOVED_LINE = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
BASELINE = INSTRUCTIONS
VARIANT = BASELINE.replace(REMOVED_LINE, "")
QUESTION = "LoRAとQLoRAの違いは？"
COURSE_QUESTION = "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-free-tier", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not args.confirm_free_tier:
        parser.error("AI Studioの現在の無料枠を確認してから --confirm-free-tier を付けてください")
    assert BASELINE.count(REMOVED_LINE) == 1
    destination = Path(__file__).parent / "results" / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("app/agent.py", "app/tools.py", "app/ask.py", "data/glossary.json", "uv.lock")}
    cases = [
        ("01_plain", COURSE_QUESTION, None),
        ("02_agent", COURSE_QUESTION, BASELINE),
        ("03_quiz", "第2回の内容から確認問題を1問出して", BASELINE),
        ("04_before", QUESTION, BASELINE),
        ("05_after", QUESTION, VARIANT),
    ]
    for name, question, instructions in cases:
        record = {"case": name, "model": MODEL, "question": question,
                  "instructions": instructions, "started_at": datetime.now().astimezone().isoformat(),
                  "source_sha256": hashes, "repeat": 1}
        print(f"Running {name}", flush=True)
        try:
            if instructions is None:
                record["output"] = ask(question)
                record["requests"] = 1
                trace = record["output"]
            else:
                target = Agent(instructions=instructions, tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz])
                result = target.run_sync(question, model=build_model(), usage_limits=UsageLimits(request_limit=6))
                record["output"] = result.output
                record["messages"] = json.loads(result.all_messages_json())
                record["requests"] = result.usage.requests
                capture = io.StringIO()
                with redirect_stdout(capture):
                    print(result.output)
                    print_trace(result)
                trace = capture.getvalue()
            record["status"] = "success"
        except Exception as exc:
            # APIエラー全文はキーや認証URLが含まれる可能性があるため保存しない。
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            (destination / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit(f"{name} failed ({type(exc).__name__}); stopped, no automatic retries") from None
        (destination / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / f"{name}.txt").write_text(trace, encoding="utf-8")
        if args.trace:
            print(trace, flush=True)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
