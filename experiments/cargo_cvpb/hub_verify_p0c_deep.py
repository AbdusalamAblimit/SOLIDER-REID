#!/usr/bin/env python3
"""P0c DEEP: is hub-mass M reducible to the trivial proxy '#false-in-topk'?

The headline original claim was 'M cleanly explains AP error beyond cheap proxies'
(partial +0.60, controlling norm/margin/camera/#pos). But the most threatening proxy
was NEVER controlled: #false-in-topk(q) = how many of q's own top-k are different-id.
This is almost tautological with AP error AND mechanically related to M (M sums H over
exactly those false neighbors). This script isolates that single control cleanly on the
FULL valid set (no NaN-driven subsetting), both raw and LOO M, both datasets.

We report, per dataset:
  rho(err, M_loo), rho(err, #false-in-topk),
  partial rho(err, M_loo | #false-in-topk)         <- THE decisive number
  partial rho(err, #false-in-topk | M_loo)         <- symmetric: which one is redundant
  Spearman(M_loo, #false-in-topk).
Also a forward check the other way: does #false-in-topk survive controlling for M?

ZERO-TRAINING: cached features + numpy.
"""
import os, sys, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--cache_feat', required=True)
ap.add_argument('--dataset', default='occluded_duke')
ap.add_argument('--k_main', type=int, default=10)
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()

def _rank(v): return np.argsort(np.argsort(v)).astype(float)
def spearman(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    x,y=x[ok],y[ok]
    if len(x)<3: return float('nan'),0
    rx,ry=_rank(x),_rank(y); rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx**2).sum()*(ry**2).sum())
    return (float((rx*ry).sum()/d) if d>0 else float('nan')), len(x)
def partial_spearman(x,y,Z):
    x=np.asarray(x,float); y=np.asarray(y,float); Z=np.asarray(Z,float)
    if Z.ndim==1: Z=Z[:,None]
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(Z).all(axis=1); x,y,Z=x[ok],y[ok],Z[ok]
    if len(x)<5: return float('nan'),0
    rx,ry=_rank(x),_rank(y)
    Zr=np.column_stack([np.ones(len(x))]+[_rank(Z[:,j]) for j in range(Z.shape[1])])
    def resid(r):
        b,*_=np.linalg.lstsq(Zr,r,rcond=None); return r-Zr@b
    ex,ey=resid(rx),resid(ry); d=np.sqrt((ex**2).sum()*(ey**2).sum())
    return (float((ex*ey).sum()/d) if d>0 else float('nan')), len(x)

def topk(sim,k):
    idx=np.argpartition(-sim,kth=k-1,axis=1)[:,:k]; rows=np.arange(sim.shape[0])[:,None]
    return idx[rows, np.argsort(-sim[rows,idx],axis=1)]

z=np.load(cli.cache_feat,allow_pickle=True)
qf=z['q_feat'].astype(np.float32); gf=z['g_feat'].astype(np.float32)
q_pid,q_cam=z['q_pid'].copy(),z['q_cam'].copy(); g_pid,g_cam=z['g_pid'].copy(),z['g_cam'].copy()
keep=g_pid!=-1; gf,g_pid,g_cam=gf[keep],g_pid[keep],g_cam[keep]
qf/=(np.linalg.norm(qf,axis=1,keepdims=True)+1e-12); gf/=(np.linalg.norm(gf,axis=1,keepdims=True)+1e-12)
Nq,Ng=qf.shape[0],gf.shape[0]; km=cli.k_main
sim=qf@gf.T
order=np.argsort(-sim,axis=1)
# per-query AP (junk removed) and raw #false-in-topk
aps=np.full(Nq,-1.0); nfalse=np.zeros(Nq)
tk=order[:,:km]
for i in range(Nq):
    oa=order[i]
    nfalse[i]=(g_pid[tk[i]]!=q_pid[i]).sum()
    keepm=~((g_pid[oa]==q_pid[i])&(g_cam[oa]==q_cam[i])); oe=oa[keepm]
    m=(g_pid[oe]==q_pid[i]).astype(np.int32)
    if not m.any(): continue
    aps[i]=((m.cumsum()/(np.arange(len(m))+1.0))*m).sum()/m.sum()
err=1.0-aps; valid=aps>=0
# H_k neg + M raw/loo
H=np.zeros(Ng,dtype=np.int64)
for c in range(km):
    gj=tk[:,c]; sel=g_pid[gj]!=q_pid; np.add.at(H,gj[sel],1)
M_raw=np.zeros(Nq); M_loo=np.zeros(Nq)
for c in range(km):
    gj=tk[:,c]; neg=g_pid[gj]!=q_pid
    M_raw+=np.where(neg,H[gj],0.0); M_loo+=np.where(neg,np.maximum(H[gj]-1,0.0),0.0)

print("#"*80); print(f"# P0c DEEP  {cli.dataset}  (n_valid={int(valid.sum())})"); print("#"*80)
rM,_=spearman(err[valid],M_loo[valid])
rF,_=spearman(err[valid],nfalse[valid])
rMrw,_=spearman(err[valid],M_raw[valid])
rHH,_=spearman(M_loo[valid],nfalse[valid])
pMrw,_=partial_spearman(err[valid],M_raw[valid],nfalse[valid])
pM,_=partial_spearman(err[valid],M_loo[valid],nfalse[valid])
pF,_=partial_spearman(err[valid],nfalse[valid],M_loo[valid])
print(f"  rho(err, M_raw)                         = {rMrw:+.4f}")
print(f"  rho(err, M_loo)                         = {rM:+.4f}")
print(f"  rho(err, #false-in-topk)                = {rF:+.4f}   <- trivial proxy")
print(f"  Spearman(M_loo, #false-in-topk)         = {rHH:+.4f}")
print(f"  partial rho(err, M_raw  | #false-in-topk) = {pMrw:+.4f}")
print(f"  partial rho(err, M_loo  | #false-in-topk) = {pM:+.4f}   <<< decisive: does hub-mass add anything?")
print(f"  partial rho(err, #false | M_loo)          = {pF:+.4f}   (symmetric: is #false still strong?)")
verdict = "REDUCIBLE to #false-in-topk (no independent value)" if (not np.isfinite(pM) or pM < 0.10) else "adds signal beyond #false-in-topk"
print(f"  >> M_loo is {verdict}")
