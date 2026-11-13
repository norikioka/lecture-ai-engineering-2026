"""保存済みDay4ログを再集計する。APIは呼ばない。"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KEYS = ("grounding", "scope", "hint", "clarity", "lecture_mark")
REQUIRED = ("grounding", "scope", "hint", "lecture_mark")


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 10 and len({r["id"] for r in rows}) == 10
    return {r["id"]: r for r in rows}


def overall(row):
    values = [row[k] for k in REQUIRED]
    return "unsure" if "unsure" in values else ("pass" if all(v == "pass" for v in values) else "fail")


def main():
    cases = read_jsonl(ROOT / "data/eval/lec04_cases.jsonl")
    labels = read_jsonl(HERE / "labels.jsonl")
    assert cases.keys() == labels.keys()
    assert all(r[k] in ("pass", "fail") for r in labels.values() for k in KEYS)
    manifests = sorted((HERE / "results").glob("*/manifest.json"))
    assert manifests, "実測manifestがない"
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["hashes_before"] == manifest["hashes_after"]
        for name, expected in manifest["hashes_before"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        assert [r["condition"] for r in manifest["runs"]] == ["before", "after"]
        assert all(r["returncode"] == 0 for r in manifest["runs"])
        for condition in ("before", "after"):
            data = json.loads((path.parent / f"{condition}.json").read_text())
            assert data["model"] == manifest["model"] != "test"
            assert data["rubric"] == f"lectures/04/experiment/rubric_{condition}.md"
            verdicts, summary = data["verdicts"], data["summary"]
            assert verdicts.keys() == cases.keys() and summary["n"] == 10
            simple = {}
            for cid, row in verdicts.items():
                assert set(row) == set(KEYS)
                assert all(v["verdict"] in ("pass", "fail", "unsure") and v["reason"].strip() for v in row.values())
                simple[cid] = {k: row[k]["verdict"] for k in KEYS}
                tail = [s.strip() for s in cases[cid]["answer"].splitlines() if s.strip()][-1]
                mark = "pass" if re.search(r"第\s*\d+\s*回", tail) else "fail"
                assert simple[cid]["lecture_mark"] == mark
            for key in KEYS:
                compared = [cid for cid in cases if simple[cid][key] != "unsure"]
                agree = sum(simple[cid][key] == labels[cid][key] for cid in compared)
                p = summary["per_criterion"][key]
                assert (p["agree"], p["compared"], p["total"]) == (agree, len(compared), 10)
            review = {(cid, k) for cid in cases for k in KEYS if simple[cid][k] == "unsure"}
            mismatch = {(cid, k) for cid in cases for k in KEYS if simple[cid][k] != "unsure" and simple[cid][k] != labels[cid][k]}
            assert review == {(v["id"], v["criterion"]) for v in summary["review"]}
            assert mismatch == {(v["id"], v["criterion"]) for v in summary["mismatches"]}
            assert len(review) == len(summary["review"]) and len(mismatch) == len(summary["mismatches"])
            for item in summary["review"]:
                assert item["reason"] == verdicts[item["id"]][item["criterion"]]["reason"]
            for item in summary["mismatches"]:
                cid, key = item["id"], item["criterion"]
                assert item["judge"] == simple[cid][key]
                assert item["human"] == labels[cid][key]
                assert item["kind"] == ("見逃し" if simple[cid][key] == "pass" else "過検出")
                assert item["reason"] == verdicts[cid][key]["reason"]
            covered = [cid for cid in cases if overall(simple[cid]) != "unsure"]
            agree = sum(overall(simple[cid]) == overall(labels[cid]) for cid in covered)
            assert summary["overall"] == {"agree": agree, "compared": len(covered)}
            assert summary["coverage"] == {"covered": len(covered), "total": 10}
            print(f"PASS {path.parent.name}/{condition}: 10件・5基準・総合・coverage・要確認・不一致")
        print("PASS 入力ハッシュ・終了コード・同一モデル・ラベル固定")


if __name__ == "__main__":
    main()
