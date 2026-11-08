import json, collections
from pathlib import Path
p=Path('data/external/raw/chembl_batch'); m=json.loads((p/'manifest.json').read_text(encoding='utf-8')); c=collections.Counter(); kinds=collections.Counter(r['kind'] for r in m['requests'])
for r in m['requests']:
 if r['kind']!='activities': continue
 d=json.loads((p/r['file']).read_text(encoding='utf-8'))
 c.update(str(x.get('standard_type')) for x in d.get('activities',[]))
print(kinds, 'assays',m.get('assay_count'),'molecules',m.get('molecule_count'))
print(c.most_common(30))
