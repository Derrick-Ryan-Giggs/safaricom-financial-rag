## usage python3 check_refusals.py

import json, re
from collections import Counter

fy_pattern = re.compile(r'FY\d{2}')
verdicts = Counter()
suspect = []

with open('answer_quality_v1.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        verdicts[rec['verdict']] += 1
        ans = rec['generated_answer']
        ans_lower = ans.lower()
        refused = 'not contain enough information' in ans_lower or 'not enough information' in ans_lower or 'do not provide' in ans_lower
        stripped = fy_pattern.sub('', ans)
        has_number = '%' in stripped or any(ch.isdigit() for ch in stripped)
        if refused and has_number:
            suspect.append(rec)

print('Verdict counts:', dict(verdicts))
print('Refusals with a real embedded number (FY-mentions excluded):', len(suspect))
for r in suspect[:5]:
    print(json.dumps(r, indent=2))
