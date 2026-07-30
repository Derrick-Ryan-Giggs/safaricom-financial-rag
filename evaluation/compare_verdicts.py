
import json
from collections import Counter

def load(path):
    by_chunk = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            by_chunk[rec['chunk_id']] = rec
    return by_chunk

v1 = load('evaluation/answer_quality_v1_backup.jsonl')
v2 = load('evaluation/answer_quality_v2_fresh.jsonl')

v1_counts = Counter(r['verdict'] for r in v1.values())
v2_counts = Counter(r['verdict'] for r in v2.values())
print('v1 verdicts:', dict(v1_counts))
print('v2 verdicts:', dict(v2_counts))

improved, regressed = [], []
rank = {'NOT_RELEVANT': 0, 'PARTLY_RELEVANT': 1, 'RELEVANT': 2}
for cid, rec1 in v1.items():
    rec2 = v2.get(cid)
    if not rec2:
        continue
    if rank[rec2['verdict']] > rank[rec1['verdict']]:
        improved.append((rec1, rec2))
    elif rank[rec2['verdict']] < rank[rec1['verdict']]:
        regressed.append((rec1, rec2))

print(f'Improved: {len(improved)}  Regressed: {len(regressed)}')
for old, new in improved[:3]:
    print('IMPROVED:', old['question'], '|', old['verdict'], '->', new['verdict'])
for old, new in regressed[:3]:
    print('REGRESSED:', old['question'], '|', old['verdict'], '->', new['verdict'])