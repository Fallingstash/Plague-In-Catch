"""Small, CPU-only neural alternatives. No pretrained weights or external labels."""
import numpy as np
import torch
from torch import nn
from rdkit import Chem
import vacv_cpu_improved as cpu
from versions.test.training import GraphNet, graph, batch, features, transform

def directed_graph(smiles):
    mol=Chem.MolFromSmiles(smiles); atoms,_=graph(smiles)
    src=[];dst=[];reverse=[];bonds=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();k=len(src)
        f=[float(b.GetBondTypeAsDouble()==x) for x in (1,2,3,1.5)]+[float(b.GetIsConjugated()),float(b.IsInRing())]
        src.extend([i,j]);dst.extend([j,i]);reverse.extend([k+1,k]);bonds.extend([f,f])
    return atoms,np.asarray(src,int),np.asarray(dst,int),np.asarray(reverse,int),np.asarray(bonds,dtype=np.float32).reshape(-1,6)

def directed_batch(graphs,ids,extra):
    atoms=[];src=[];dst=[];rev=[];bonds=[];membership=[];aoff=0;eoff=0
    for j,i in enumerate(ids):
        a,s,d,r,b=graphs[i];atoms.append(a);src.append(s+aoff);dst.append(d+aoff);rev.append(r+eoff);bonds.append(b);membership.extend([j]*len(a));aoff+=len(a);eoff+=len(s)
    return (torch.tensor(np.concatenate(atoms),dtype=torch.float32),torch.tensor(np.concatenate(src),dtype=torch.long),torch.tensor(np.concatenate(dst),dtype=torch.long),torch.tensor(np.concatenate(rev),dtype=torch.long),torch.tensor(np.concatenate(bonds),dtype=torch.float32),torch.tensor(membership,dtype=torch.long),torch.from_numpy(extra[ids]))

class DirectedMPNN(nn.Module):
    def __init__(self,extra_dim,width=32,depth=2):
        super().__init__();self.depth=depth;self.edge=nn.Linear(16,width);self.message=nn.Linear(width,width);self.atom=nn.Linear(10+width,width);self.head=nn.Sequential(nn.Linear(width+extra_dim,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,atoms,src,dst,reverse,bonds,membership,extra):
        h0=torch.relu(self.edge(torch.cat([atoms[src],bonds],dim=1)));h=h0
        for _ in range(self.depth-1):
            incoming=atoms.new_zeros((len(atoms),h.shape[1]));incoming.index_add_(0,dst,h)
            h=torch.relu(h0+self.message(incoming[src]-h[reverse]))
        incoming=atoms.new_zeros((len(atoms),h.shape[1]));incoming.index_add_(0,dst,h)
        atom=torch.relu(self.atom(torch.cat([atoms,incoming],dim=1)))
        pooled=atoms.new_zeros((len(extra),atom.shape[1]));pooled.index_add_(0,membership,atom)
        counts=torch.bincount(membership,minlength=len(extra)).clamp_min(1)
        pooled=pooled/counts[:,None]
        return self.head(torch.cat([pooled,extra],dim=1)).squeeze(1)

class ResidualMLP(nn.Module):
    def __init__(self,width):
        super().__init__();self.project=nn.Sequential(nn.Linear(width,128),nn.LayerNorm(128),nn.GELU());self.block=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.2));self.head=nn.Linear(128,1)
    def forward(self,x):
        z=self.project(x);return self.head(z+self.block(z)).squeeze(1)

def bond_graph(smiles):
    atoms,_=graph(smiles);mol=Chem.MolFromSmiles(smiles);a=np.eye(len(atoms),dtype=np.float32)
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();a[i,j]=a[j,i]=bond.GetBondTypeAsDouble()
    degree=a.sum(1);return atoms,a/np.sqrt(degree[:,None]*degree[None,:])

class NeuralRegressor:
    def __init__(self,mode='gcn_mean3',epochs=80):self.mode=mode;self.epochs=epochs
    def fit(self,frame,y,seed=42):
        self.training_rows=len(y);cpu.set_seed(seed)
        raw=cpu.chem_features(frame.SMILES.tolist());ids=np.arange(len(y));extra,self.scaler=features(raw,ids)
        make=directed_graph if self.mode=='dmpnn' else bond_graph if self.mode=='bond_gcn' else graph
        graphs=[make(s) for s in frame.SMILES]
        self.models=[]
        for run in range(3 if self.mode=='gcn_mean3' else 1):
            init=seed+run*1000;cpu.set_seed(init)
            model=DirectedMPNN(extra.shape[1]) if self.mode=='dmpnn' else ResidualMLP(extra.shape[1]) if self.mode=='residual_mlp' else GraphNet(extra.shape[1])
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(init);yy=torch.tensor(y,dtype=torch.float32)
            for _ in range(self.epochs):
                model.train()
                for b in np.array_split(rng.permutation(ids),max(1,int(np.ceil(len(ids)/32)))):
                    if self.mode=='dmpnn':out=model(*directed_batch(graphs,b,extra))
                    elif self.mode=='residual_mlp':out=model(torch.from_numpy(extra[b]))
                    else:out=model(*batch(graphs,b,extra))
                    loss=nn.functional.mse_loss(out,torch.log1p(yy[b]))
                    if self.mode=='hybrid_gcn':loss=loss+.1*nn.functional.mse_loss(torch.expm1(out.clamp(0,14))/5000,yy[b]/5000)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step()
            model.eval();self.models.append(model)
        return self
    def predict(self,frame):
        raw=cpu.chem_features(frame.SMILES.tolist());extra=transform(raw,self.scaler)
        make=directed_graph if self.mode=='dmpnn' else bond_graph if self.mode=='bond_gcn' else graph
        graphs=[make(s) for s in frame.SMILES];ids=np.arange(len(frame));result=[]
        with torch.no_grad():
            for model in self.models:
                p=[];model.eval()
                for b in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
                    if self.mode=='dmpnn':out=model(*directed_batch(graphs,b,extra))
                    elif self.mode=='residual_mlp':out=model(torch.from_numpy(extra[b]))
                    else:out=model(*batch(graphs,b,extra))
                    p.extend(torch.expm1(out.clamp(0,14)).numpy())
                result.append(p)
        return np.mean(result,axis=0).astype(float)
