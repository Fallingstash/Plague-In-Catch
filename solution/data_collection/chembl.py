"""Download public ChEMBL VACV assay records for manual review.

Searches by organism, never by competition test molecules. Raw assay, activity,
molecule and document API responses are kept with source URLs. Records are NOT
approved training labels: assay context and paired measurements need review.
"""
import argparse, json, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE='https://www.ebi.ac.uk/chembl/api/data/'
HEADERS={'User-Agent':'VACV-QSAR-research/1.0 (public ChEMBL API)','Accept':'application/json'}

def get_json(url,timeout=30,retries=2):
 for attempt in range(retries+1):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=timeout) as response:
    return json.load(response)
  except (urllib.error.URLError,TimeoutError) as exc:
   if attempt==retries: raise RuntimeError(f'Could not retrieve {url}: {exc}') from exc
   time.sleep(2**attempt)

def paginate(endpoint,params,record_key,max_pages):
 url=BASE+endpoint+'.json?'+urllib.parse.urlencode({**params,'limit':100})
 for page in range(max_pages):
  data=get_json(url); yield url,data
  next_url=data.get('page_meta',{}).get('next')
  if not next_url: return
  url=urllib.parse.urljoin(BASE,next_url)
 raise RuntimeError(f'More than {max_pages} pages; incomplete download. Increase --max-pages.')

def chunks(items,size=40):
 items=sorted(items)
 for i in range(0,len(items),size): yield items[i:i+size]

def collect(out,max_pages):
 out=Path(out)
 if out.exists() and any(out.iterdir()): raise ValueError(f'Output directory is not empty: {out}; use a new directory')
 out.mkdir(parents=True,exist_ok=True)
 manifest={'source':'ChEMBL REST API','api_documentation':'https://www.ebi.ac.uk/chembl/api/data/docs','query':'assay organism contains Vaccinia', 'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'incomplete','requests':[]}
 def save(kind,url,data):
  path=out/f'{len(manifest["requests"]):05d}_{kind}.json'; path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
  manifest['requests'].append({'kind':kind,'url':url,'file':path.name}); (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 try:
  assays=[]
  for url,data in paginate('assay',{'assay_organism__icontains':'vaccinia'},'assays',max_pages):
   save('assays',url,data); assays.extend(data.get('assays',[]))
  molecule_ids=set(); document_ids=set()
  assay_ids={a['assay_chembl_id'] for a in assays if a.get('assay_chembl_id')}
  for batch in chunks(assay_ids):
   for url,data in paginate('activity',{'assay_chembl_id__in':','.join(batch)},'activities',max_pages):
    save('activities',url,data)
    for act in data.get('activities',[]):
     if act.get('molecule_chembl_id'): molecule_ids.add(act['molecule_chembl_id'])
     if act.get('document_chembl_id'): document_ids.add(act['document_chembl_id'])
  # Retrieve molecules, documents, and all cytotoxicity measurements from the same papers.
  toxicity_assay_ids=set()
  for batch in chunks(document_ids):
   for url,data in paginate('document',{'document_chembl_id__in':','.join(batch)},'documents',max_pages): save('documents',url,data)
   for url,data in paginate('activity',{'document_chembl_id__in':','.join(batch),'standard_type':'CC50'},'activities',max_pages):
    save('cc50',url,data)
    toxicity_assay_ids.update(x['assay_chembl_id'] for x in data.get('activities',[]) if x.get('assay_chembl_id'))
  for batch in chunks(toxicity_assay_ids):
   for url,data in paginate('assay',{'assay_chembl_id__in':','.join(batch)},'assays',max_pages): save('cc50_assays',url,data)
  for batch in chunks(molecule_ids):
   for url,data in paginate('molecule',{'molecule_chembl_id__in':','.join(batch)},'molecules',max_pages): save('molecules',url,data)
  manifest['status']='complete'; manifest['assay_count']=len(assays); manifest['molecule_count']=len(molecule_ids); manifest['document_count']=len(document_ids)
 except Exception as exc:
  manifest['error']=str(exc); raise
 finally:
  (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Fetched {len(assays)} assays, {len(molecule_ids)} molecules and {len(document_ids)} documents. Review all measurements before use.')

def supplement_toxicity_assays(out,max_pages):
 out=Path(out); manifest_path=out/'manifest.json'; manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 if manifest.get('status')!='complete': raise ValueError('Cannot supplement an incomplete download')
 existing=set(); needed=set()
 for entry in manifest['requests']:
  if entry['kind'] not in ('assays','cc50','cc50_assays'): continue
  payload=json.loads((out/entry['file']).read_text(encoding='utf-8'))
  if entry['kind']=='cc50': needed.update(x['assay_chembl_id'] for x in payload.get('activities',[]) if x.get('assay_chembl_id'))
  else: existing.update(x['assay_chembl_id'] for x in payload.get('assays',[]) if x.get('assay_chembl_id'))
 for batch in chunks(needed-existing):
  for url,data in paginate('assay',{'assay_chembl_id__in':','.join(batch)},'assays',max_pages):
   path=out/f'{len(manifest["requests"]):05d}_cc50_assays.json'; path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
   manifest['requests'].append({'kind':'cc50_assays','url':url,'file':path.name}); manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Toxicity assay metadata: {len(needed)} referenced assays; {len(needed-existing)} newly requested.')

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',default='data/external/raw/chembl'); p.add_argument('--max-pages',type=int,default=100); p.add_argument('--supplement-cc50-assays',action='store_true'); a=p.parse_args()
 if a.supplement_cc50_assays: supplement_toxicity_assays(a.out,a.max_pages)
 else: collect(a.out,a.max_pages)
if __name__=='__main__': main()
