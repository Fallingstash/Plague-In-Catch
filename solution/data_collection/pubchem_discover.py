"""Discover PubChem BioAssay VACV records and download candidate SI assays."""
import argparse,csv,json,time,urllib.parse,urllib.request
from pathlib import Path
from competition_data import require_safe_output

BASE='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'VACV-data-research/1.0'}),timeout=60) as r: return r.read()
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True); p.add_argument('--max-download',type=int,default=50); a=p.parse_args()
    if a.max_download<0: p.error('--max-download must be nonnegative')
    out=require_safe_output(a.out); out.mkdir(parents=True,exist_ok=True)
    query='vaccinia[All Fields]'
    search=BASE+'esearch.fcgi?'+urllib.parse.urlencode({'db':'pcassay','term':query,'retmode':'json','retmax':9999,'tool':'vacv_research'})
    payload=json.loads(get(search)); result=payload['esearchresult']; ids=result['idlist']
    (out/'search.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if int(result['count'])>len(ids): raise ValueError('Search results truncated; partition query before continuing')
    rows=[]
    for start in range(0,len(ids),200):
        url=BASE+'esummary.fcgi?'+urllib.parse.urlencode({'db':'pcassay','id':','.join(ids[start:start+200]),'retmode':'json','tool':'vacv_research'})
        path=out/f'summary_{start:05d}.json'
        if not path.exists(): path.write_bytes(get(url))
        data=json.loads(path.read_text(encoding='utf-8'))['result']
        for uid in data['uids']:
            item=data[uid]; name=item.get('assayname',''); desc=item.get('assaydescription',''); text=(name+' '+desc).lower()
            is_si=('selectivity index' in text or 'selectivity indices' in text)
            rows.append({'aid':uid,'name':name,'description':desc,'source':'; '.join(item.get('sourcenamelist',[])),'cell_line':item.get('cellline',''),'total_substances':item.get('totalsidcount',''),'candidate_si':is_si,'url':'https://pubchem.ncbi.nlm.nih.gov/bioassay/'+uid})
        print(f'Discovered {len(rows)}/{len(ids)} assay summaries',flush=True); time.sleep(.4)
    with (out/'assay_index.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['aid','name','description','source','cell_line','total_substances','candidate_si','url']); w.writeheader();w.writerows(rows)
    candidates=[r for r in rows if 'vaccinia' in r['name'].lower() and 'selectiv' in r['name'].lower()]
    downloads=[]
    for row in candidates[:a.max_download]:
        aid=row['aid']; dest=out/f'assay_{aid}.csv'; url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV'
        try:
            if not dest.exists(): dest.write_bytes(get(url))
            downloads.append({'aid':aid,'url':url,'file':dest.name,'status':'downloaded'})
        except Exception as exc:
            downloads.append({'aid':aid,'url':url,'status':'error','error':str(exc)})
        print(f"PubChem {aid}: {downloads[-1]['status']}",flush=True); time.sleep(.4)
    (out/'manifest.json').write_text(json.dumps({'source':'PubChem','search_url':search,'total_search_hits':int(result['count']),'summaries':len(rows),'si_candidates':len(candidates),'downloads':downloads,'warning':'Search matches can concern vaccinia as a reagent or another virus. Raw assay columns, origin, license and endpoint require review; not training labels.'},indent=2),encoding='utf-8')
    print(f'Finished: {len(rows)} summaries, {len(candidates)} candidate SI assays, {len(downloads)} download attempts',flush=True)
if __name__=='__main__':main()

