"""APIを使わず、原メッセージから12実行の数値と固定条件を検算する。"""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / "experiment_plan.json").read_text())
    expected = {f"k{k}_{q['id']}_r{r}" for k in plan["top_k"]
                for q in plan["questions"] for r in range(1, plan["repetitions"] + 1)}
    files = list((HERE / "results").glob("k*.json"))
    assert {p.stem for p in files} == expected, "実測12件の不足・重複"
    rows = []
    for path in sorted(files):
        d = json.loads(path.read_text())
        assert d['run_id'] == path.stem
        assert path.stem == f"k{d['top_k']}_{d['question_id']}_r{d['repetition']}"
        assert d['model'] == plan['model'] and d['embedding_model'] == plan['embedding_model']
        q = next(q for q in plan['questions'] if q['id'] == d['question_id'])
        assert d['question'] == q['question']
        assert d['effective_db'] == 'data/rag_data/index.sqlite3'
        expected_paths = {'app/config.py', 'app/agent.py', 'app/tools.py', 'app/retrieval.py',
                          'data/rag_data/lectures.json', 'data/rag_data/index.sqlite3', 'uv.lock',
                          'lectures/05/experiment_plan.json', 'lectures/05/run_capture.py'}
        assert set(d['sha256']) == expected_paths, 'hash記録の欠落・余剰'
        for name, digest in d['sha256'].items():
            data = (ROOT / name).read_bytes()
            if name == 'app/config.py':
                # 現在の提出コードがk3であることと、実行時の唯一の差分を検算。
                assert data.count(b'RAG_TOP_K = 3\n') == 1
                data = data.replace(b'RAG_TOP_K = 3\n', f"RAG_TOP_K = {d['top_k']}\n".encode())
            assert hashlib.sha256(data).hexdigest() == digest, f'{path.name}: hash {name}'
        messages = d['messages']
        responses = [m for m in messages if m['kind'] == 'response']
        parts = [p for m in messages for p in m['parts']]
        calls = [p for p in parts if p['part_kind'] == 'tool-call']
        returns = [p for p in parts if p['part_kind'] == 'tool-return']
        prompts = [p['content'] for p in parts if p['part_kind'] == 'user-prompt']
        assert prompts == [q['question']]
        assert all(m['model_name'] == plan['model'] for m in responses)
        assert len(responses) == d['usage']['requests']
        for key in ['input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens']:
            assert sum(m['usage'][key] for m in responses) == d['usage'][key], key
        assert len(calls) == d['usage']['tool_calls'] and len(calls) > 0
        assert {p['tool_call_id'] for p in calls} == {p['tool_call_id'] for p in returns}
        assert all(p['tool_name'] == 'search_materials' for p in calls + returns)
        assert d['elapsed_seconds'] > 0 and d['output'].strip()
        assert ''.join(p['content'] for p in responses[-1]['parts'] if p['part_kind'] == 'text') == d['output']
        # SDKの集計costと保存した表示用costの整合性（独立した請求額検証ではない）。
        cost = d['estimated_generation_cost_usd']
        if cost is None:
            assert d['usage']['cost'] is None
        else:
            assert Decimal(cost) == Decimal(str(d['usage']['cost']))
        rows.append({'run':d['run_id'], 'seconds':round(d['elapsed_seconds'],2),
                     'input':d['usage']['input_tokens'], 'output':d['usage']['output_tokens'],
                     'requests':len(responses), 'tool_calls':len(calls), 'cost':cost})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print('PASS: 12実行の条件・ハッシュ・原メッセージ・usage照合（回答内容の正しさは別途レビュー）')


if __name__ == '__main__':
    main()
