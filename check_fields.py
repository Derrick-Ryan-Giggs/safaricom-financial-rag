import json, glob

for path in sorted(glob.glob('embeddings/*.jsonl')):
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('fiscal_year') == 'FY09' and 'Revenue 70,480' in rec.get('text', ''):
                rec.pop('embedding', None)
                print(json.dumps(rec, indent=2, default=str))
