"""Optional structure lookup only; no competition activity reconstruction."""
import json
import urllib.request

def structure_by_cid(cid):
    cid=int(cid)
    if cid<=0:raise ValueError('Positive PubChem CID required')
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON'
    with urllib.request.urlopen(url,timeout=30) as response:
        record=json.load(response)
    return {'source_url':url,'raw_record':record,'training_allowed':False}
