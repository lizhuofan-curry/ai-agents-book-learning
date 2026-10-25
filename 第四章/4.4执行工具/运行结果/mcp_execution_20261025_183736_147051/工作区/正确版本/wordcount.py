from collections import Counter
import json
from pathlib import Path

words = Path('data.txt').read_text(encoding='utf-8').split()
counts = dict(Counter(words))
Path('counts.json').write_text(json.dumps(counts, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps(counts, sort_keys=True))
