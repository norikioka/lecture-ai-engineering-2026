"""教材の --trace 実行を保ち、回答・全ツール結果・usageをJSONへ保存する。"""
import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app import agent as module
from app import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question_id", choices=["q1", "q2", "q3"])
    parser.add_argument("repetition", type=int, choices=[1, 2])
    args = parser.parse_args()
    plan_path = Path(__file__).with_name("experiment_plan.json")
    plan = json.loads(plan_path.read_text())
    question = next(q["question"] for q in plan["questions"] if q["id"] == args.question_id)
    if (config.MODEL != plan["model"] or config.RAG_TOP_K not in plan["top_k"]
            or config.EMBEDDING_MODEL != plan["embedding_model"]
            or config.RAG_DB.resolve() != (ROOT / "data/rag_data/index.sqlite3").resolve()):
        raise SystemExit("固定したモデルまたは検索件数と不一致")
    stem = f"k{config.RAG_TOP_K}_{args.question_id}_r{args.repetition}"
    output = Path(__file__).with_name("results") / f"{stem}.json"
    if output.exists():
        raise SystemExit(f"上書き禁止: {output}")
    original = module.print_trace

    def capture(result, elapsed_seconds=None):
        original(result, elapsed_seconds)
        paths = ["app/config.py", "app/agent.py", "app/tools.py", "app/retrieval.py",
                 "data/rag_data/lectures.json", "data/rag_data/index.sqlite3", "uv.lock",
                 "lectures/05/experiment_plan.json", "lectures/05/run_capture.py"]
        usage = dataclasses.asdict(result.usage)
        usage["cost"] = None if result.usage.cost is None else str(result.usage.cost)
        record = {
            "run_id": stem, "completed_at": datetime.now().astimezone().isoformat(),
            "question_id": args.question_id, "question": question,
            "repetition": args.repetition, "top_k": config.RAG_TOP_K,
            "model": config.MODEL, "embedding_model": config.EMBEDDING_MODEL,
            "elapsed_seconds": elapsed_seconds, "output": result.output,
            "usage": usage,
            "effective_db": str(config.RAG_DB.resolve().relative_to(ROOT)),
            "estimated_generation_cost_usd": None if result.usage.cost is None else str(result.usage.cost),
            "messages": json.loads(result.all_messages_json()),
            "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        }
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        print(f"[saved] {output.relative_to(ROOT)}")

    module.print_trace = capture
    sys.argv = ["app.agent", "--trace", question]
    module.main()


if __name__ == "__main__":
    main()
