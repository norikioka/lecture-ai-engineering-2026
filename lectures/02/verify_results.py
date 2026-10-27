"""保存済みの生出力を再採点し、2条件の比較可能性を検証する（GPU不要）。"""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def score(raw, gold):
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    fmt = (isinstance(obj, dict) and set(obj) == {'answer', 'needs_review'}
           and isinstance(obj['answer'], str) and isinstance(obj['needs_review'], bool))
    if not fmt:
        match = re.search(r'\{.*\}', raw, re.S)
        try:
            obj = json.loads(match.group()) if match else None
        except ValueError:
            obj = None
    content = (isinstance(obj, dict) and obj.get('needs_review') == gold['needs_review']
               and all(w in obj.get('answer', '') for w in gold['must_include']))
    return fmt, content

results = []
evals = {}
for epochs in (20, 10):
    folder = ROOT / 'results' / f'epochs{epochs}'
    result = read(folder / 'result.json')
    assert result['config']['num_train_epochs'] == epochs
    assert result['executed_training_myself'] is True
    assert 'Tesla T4' in result['gpu']
    train = result['training']
    assert train['epoch'] == epochs and train['global_step'] == epochs * 2
    assert math.isfinite(train['seconds']) and train['seconds'] > 0
    last = [r for r in train['log_history'] if 'loss' in r][-1]
    assert last['step'] == train['final_loss_step'] == epochs * 2
    assert last['loss'] == train['final_logged_loss'] and math.isfinite(last['loss'])
    for name, digest in result['data_sha256'].items():
        assert name in ('test.jsonl', 'gold.json')
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    gold = read(folder / 'gold.json')
    questions = [json.loads(line)['question'] for line in (folder / 'test.jsonl').read_text().splitlines()]
    assert len(questions) == len(set(questions)) == len(gold) == 21
    for label in ('before', 'after'):
        evaluation = read(folder / f'eval_{label}.json')
        rows = evaluation['rows']
        assert [r['question'] for r in rows] == questions
        for row in rows:
            fmt, content = score(row['raw'], gold[row['question']])
            assert (fmt, content, fmt and content) == (row['format_ok'], row['content_ok'], row['both_ok'])
            assert row['gold_needs_review'] == gold[row['question']]['needs_review']
        assert evaluation['summary'] == result[label]
        assert result[label]['n'] == 21
        for metric, key in [('format_rate', 'format_ok'), ('content_rate', 'content_ok'), ('both_rate', 'both_ok')]:
            assert math.isclose(sum(r[key] for r in rows) / 21, result[label][metric])
        evals[epochs, label] = rows
    regression = [b['question'] for b, a in zip(evals[epochs, 'before'], evals[epochs, 'after']) if b['both_ok'] and not a['both_ok']]
    assert regression == result['regressed']
    results.append(result)
    print(f'PASS: epochs={epochs}, step={train["global_step"]}, raw outputs rescored, hashes/loss/regression verified')
a, b = results
assert {k: v for k, v in a['config'].items() if k != 'num_train_epochs'} == {k: v for k, v in b['config'].items() if k != 'num_train_epochs'}
for key in ('versions', 'gpu', 'python', 'source_notebook_sha256', 'data_sha256'):
    assert a[key] == b[key], key
assert evals[20, 'before'] == evals[10, 'before']
print('PASS: configs differ only in epochs; recorded environment, evaluation data and all baseline outputs agree')
print('Training-time reduction: %.2f%%' % ((1 - b['training']['seconds'] / a['training']['seconds']) * 100))
