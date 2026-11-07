"""SVA node attribution, independent native checks, and explicit q interventions.

The default unit is a neuron at a position. Experts are sums over ALL neurons,
before ranking. This module never changes the historical FModel/pilot results.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read_split(root, task, split):
    root=Path(root)
    meta=json.loads((root/'manifest.json').read_text())
    path=root/task/f'{split}.jsonl'
    if sha(path)!=meta['tasks'][task]['files_sha256'][split]:
        raise ValueError(f'Manifest checksum mismatch: {path}')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f'Empty split: {split}')
    return meta,rows


def identity(fm, model_dir):
    import transformers
    model_dir=Path(model_dir).resolve()
    # Bind acceptance to this exact local checkpoint installation and runtime.
    files=[p for p in sorted(model_dir.glob('*')) if p.is_file() and
           (p.suffix in ('.json','.safetensors','.bin','.model'))]
    return {'model_dir':str(model_dir),'dtype':str(fm.dtype),'device':str(fm.dev),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'fmodel_sha256':sha(Path(__file__).with_name('ext14_circuit_relp.py')),
        'sva_sha256':sha(__file__),
        'files':{p.name:({'sha256':sha(p)} if p.suffix=='.json' else
                         {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}) for p in files}}


def check_tokenizer_files(model_dir, meta):
    for name,expected in meta['tokenizer_sha256'].items():
        # config also pins the architecture. A distinct model needs its own manifest.
        p=Path(model_dir)/name
        if not p.exists() or sha(p)!=expected:
            raise ValueError(f'Model/tokenizer differs from prepared manifest: {name}')


def tensor(fm, row, reference=False):
    return torch.tensor([row['reference_ids' if reference else 'input_ids']],device=fm.dev)


def position_mask(row, pool):
    mask=torch.tensor([not x for x in row['special_tokens_mask']],dtype=torch.bool)
    if pool=='all':mask[:]=True
    elif pool=='final':mask[:]=False;mask[-1]=True
    elif pool!='non_special':raise ValueError(pool)
    return mask


def score_pair(fm, row, mode='paired', pool='non_special'):
    target,foil=[row['target_id']],[row['foil_id']]
    reference=None
    if mode=='paired':
        with torch.no_grad():
            r=fm.forward(tensor(fm,row,True),target,foil,record=True)
            reference=[x[0].float().cpu() for x in r['q']]
            ref_metric=float(r['delta'][0])
            del r
    else:ref_metric=None
    with torch.enable_grad():
        main=fm.forward(tensor(fm,row),target,foil,relp=True,grad=True)
        beta=torch.autograd.grad(main['delta'].sum(),main['q'])
    scores,support=[],[]
    for l,(q,g) in enumerate(zip(main['q'],beta)):
        q=q[0].detach().float().cpu().reshape(len(row['input_ids']),fm.E,fm.I)
        g=g[0].detach().float().cpu().reshape_as(q)
        ref=reference[l].reshape_as(q) if reference is not None else torch.zeros_like(q)
        score=(q-ref)*g
        if not torch.isfinite(score).all():raise ValueError('Nonfinite attribution')
        pm=position_mask(row,pool)[:,None,None]
        scores.append(torch.where(pm,score,0))
        support.append(((q!=0)|(ref!=0))&pm)
    return torch.stack(scores),torch.stack(support),{
        'row_id':row['row_id'],'group_id':row['group_id'],'margin_A':float(main['delta'][0].detach()),
        'margin_B':ref_metric,'endpoint':'A','mode':mode}


def rank_indices(values, eligible, n):
    flat=values.reshape(-1)
    candidates=torch.where(eligible.reshape(-1))[0]
    n=min(n,len(candidates))
    if n<=0:return torch.empty(0,dtype=torch.long)
    # Stable tie break by flat coordinate, including boundary ties.
    local=torch.argsort(flat[candidates].abs(),descending=True,stable=True)[:n]
    return candidates[local]


def coordinates(indices, shape):
    return [list(map(int,np.unravel_index(i,shape))) for i in indices.tolist()]


def rankings(mean, support, top_k=10000):
    out={}
    for unit,values,eligible in (
        ('neuron',mean,support),
        ('expert_position',mean.sum(-1),support.any(-1)),
        ('expert_all_positions',mean.sum(-1).sum(1),support.any(-1).any(1))):
        ix=rank_indices(values,eligible,top_k if unit=='neuron' else values.numel())
        out[unit]=[{'coordinate':c,'score':float(values.reshape(-1)[i]),'rank':rank+1}
                   for rank,(i,c) in enumerate(zip(ix.tolist(),coordinates(ix,values.shape)))]
    return out


def native_capture(fm, ids, target, foil, output_patch=None):
    """HF forward hooks; weighted q reconstructed from native selected experts/inputs.

    This is independent of FModel's batched all-expert einsum. Native expert
    routing is read from the actual module call, never from a second router pass.
    """
    store={'out':{},'q':{},'route':{}}
    handles=[]
    for l,layer in enumerate(fm.model.model.layers):
        def out_hook(mod,args,out,l=l):
            store['out'][l]=out.detach()
            if output_patch is not None and l in output_patch:
                mask,value=output_patch[l]
                return torch.where(mask[...,None],value.to(out),out)
        handles.append(layer.mlp.register_forward_hook(out_hook))
        if fm.moe:
            def experts_pre(mod,args,l=l):
                x,indices,weights=args
                n=x.shape[0]
                q=x.new_zeros(n,fm.E,fm.I)
                for e in range(fm.E):
                    row,slot=torch.where(indices==e)
                    if row.numel():
                        g,u=F.linear(x[row],mod.gate_up_proj[e]).chunk(2,-1)
                        q[row,e]=F.silu(g)*u*weights[row,slot,None]
                store['q'][l]=q.reshape(ids.shape[0],ids.shape[1],fm.E,fm.I)
                store['route'][l]=(indices.detach(),weights.detach())
            handles.append(layer.mlp.experts.register_forward_pre_hook(experts_pre))
        else:
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda m,a,l=l:store['q'].__setitem__(l,a[0].detach())))
    handles.append(fm.model.model.norm.register_forward_pre_hook(
        lambda m,a:store.__setitem__('residual',a[0].detach())))
    try:
        with torch.no_grad():
            normed=fm.model.model(ids,use_cache=False,return_dict=True).last_hidden_state
            u=fm.W_U[target].float()-fm.W_U[foil].float()
            store['delta']=(normed[:,-1].float()*u).sum(-1)
    finally:
        for h in handles:h.remove()
    return store


def verify(fm, rows, atol, rtol, max_relative):
    checks=[]
    def compare(name,a,b):
        a,b=a.detach().float(),b.detach().float()
        diff=a-b
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        rel=float(diff.norm()/b.norm().clamp_min(atol)) if finite else None
        checks.append({'name':name,'max_abs':float(diff.abs().max()) if finite else None,
            'relative_l2':rel,'passed':finite and bool(torch.allclose(a,b,atol=atol,rtol=rtol))
            and rel<=max_relative})
    first=None
    for row in rows:
        for reference in (False,True):
            ids=tensor(fm,row,reference)
            hf=native_capture(fm,ids,[row['target_id']],[row['foil_id']])
            with torch.no_grad():
                plain=fm.forward(ids,[row['target_id']],[row['foil_id']],record=True)
                relp=fm.forward(ids,[row['target_id']],[row['foil_id']],record=True,relp=True)
            tag=row['row_id']+('/B' if reference else '/A')
            compare(tag+'/HF/metric',plain['delta'],hf['delta'])
            compare(tag+'/HF/final_residual',plain['h_final'],hf['residual'][:,-1])
            compare(tag+'/RelP/metric',relp['delta'],plain['delta'])
            for l in range(fm.L):
                compare(f'{tag}/HF/L{l}/out',plain['out'][l],hf['out'][l])
                compare(f'{tag}/HF/L{l}/q',plain['q'][l],hf['q'][l])
                compare(f'{tag}/RelP/L{l}/q',relp['q'][l],plain['q'][l])
                if fm.moe:
                    pi,pw=plain['route'][l];hi,hw=hf['route'][l]
                    # Expert identity, not top-k slot order.
                    ps,po=pi.sort(-1);hs,ho=hi.sort(-1)
                    checks.append({'name':f'{tag}/HF/L{l}/topk','passed':bool(torch.equal(ps,hs))})
                    compare(f'{tag}/HF/L{l}/weights',pw.gather(-1,po),hw.gather(-1,ho))
            if first is None:first=(row,plain,hf)
        _,grads=fm.node_grads(tensor(fm,row),[row['target_id']],[row['foil_id']],True)
        checks.append({'name':row['row_id']+'/natural_A/finite_beta',
                       'passed':all(bool(torch.isfinite(g).all()) for g in grads)})
    # Actual checkpoint: all-neuron patch equals output patch, and agrees with a native HF output hook.
    row,clean,hf_clean=first
    ids=tensor(fm,row,True);target=[row['target_id']];foil=[row['foil_id']]
    for l in sorted({0,fm.L//2,fm.L-1}):
        mask=torch.zeros_like(clean['q'][l],dtype=torch.bool);mask[:,-1]=True
        bt=torch.zeros_like(ids,dtype=torch.bool);bt[:,-1]=True
        with torch.no_grad():
            qpatch=fm.forward(ids,target,foil,interv={l:(mask,clean['q'][l])})
            opatch=fm.forward(ids,target,foil,out_patch={l:(bt,clean['out'][l])})
        native=native_capture(fm,ids,target,foil,{l:(bt,hf_clean['out'][l])})
        compare(f'patch/L{l}/all_q_vs_out',qpatch['delta'],opatch['delta'])
        compare(f'patch/L{l}/functional_vs_native',qpatch['delta'],native['delta'])
    return {'passed':all(c['passed'] for c in checks),'checks':checks,
            'tolerances':{'atol':atol,'rtol':rtol,'max_relative_l2':max_relative},
            'rows':[r['row_id'] for r in rows],'scope':'short calibration pairs; native q/out/routes and whole-output interventions'}


def random_matched(selected, eligible, generator):
    """Match exact layer/position counts within discovery-active candidates."""
    flat=eligible.reshape(eligible.shape[0],eligible.shape[1],-1)
    chosen=torch.zeros_like(flat)
    selected=selected.reshape_as(flat)
    for l in range(flat.shape[0]):
        for t in range(flat.shape[1]):
            n=int(selected[l,t].sum())
            if n:
                candidates=torch.where(flat[l,t])[0]
                chosen[l,t,candidates[torch.randperm(len(candidates),generator=generator)[:n]]]=True
    return chosen.reshape_as(eligible)


def calibrate_q(fm, rows):
    sums=counts=None
    with torch.no_grad():
        for row in rows:
            r=fm.forward(tensor(fm,row),[row['target_id']],[row['foil_id']],record=True)
            qs=torch.stack([x[0].float().cpu().reshape(len(row['input_ids']),fm.E,fm.I) for x in r['q']])
            active=torch.ones(qs.shape[:-1],dtype=torch.float64)
            if fm.moe:
                active.zero_()
                for l,(indices,_) in enumerate(r['route']):active[l].scatter_(1,indices.cpu(),1)
            if sums is None:sums=torch.zeros_like(qs,dtype=torch.float64);counts=torch.zeros_like(active)
            sums+=qs.double();counts+=active
    return (sums/counts.clamp_min(1)[...,None]).float(),counts


def route_change(before, after):
    sets=0;weight_l1=0.
    for (ai,aw),(bi,bw) in zip(before,after):
        a,ap=ai.sort(-1);b,bp=bi.sort(-1)
        sets+=int((a!=b).any(-1).sum())
        # Match by ID even when a set changes.
        e=int(torch.maximum(ai.max(),bi.max()))+1
        ga=aw.new_zeros(aw.shape[0],e).scatter(1,ai,aw)
        gb=bw.new_zeros(bw.shape[0],e).scatter(1,bi,bw)
        weight_l1+=float((ga-gb).abs().sum())
    return {'topk_changed_layer_positions':sets,'routing_weight_l1':weight_l1}


def evaluate_case(fm,row,masks,baseline,pool,qmean=None):
    """One fixed discovered list, evaluated on this held-out row with natural downstream routing."""
    target,foil=[row['target_id']],[row['foil_id']]
    with torch.no_grad():
        a=fm.forward(tensor(fm,row),target,foil,record=True)
        b=fm.forward(tensor(fm,row,True),target,foil,record=True)
        pm=position_mask(row,pool).to(fm.dev)
        def forward(mask,kind):
            interventions={}
            for l in range(fm.L):
                ml=mask[l].to(fm.dev).reshape_as(a['q'][l])
                p=pm.view(1,-1,*([1]*(ml.ndim-2)))
                if kind in ('keep','remove','empty'):
                    sel=(~ml if kind=='keep' else ml)
                    sel=sel&p
                    values=torch.zeros_like(a['q'][l]) if qmean is None else qmean[l].to(fm.dev).reshape_as(a['q'][l])
                    interventions[l]=(sel,values,'routed') if baseline=='mean_q_routed' and fm.moe else (sel,values)
                else:
                    interventions[l]=(ml&p,b['q'][l] if kind=='noise' else a['q'][l])
            return fm.forward(tensor(fm,row,kind=='denoise'),target,foil,interv=interventions)
        shape=(fm.L,len(row['input_ids']),fm.E,fm.I)
        empty=forward(torch.ones(shape,dtype=torch.bool),'empty')
        common={'row_id':row['row_id'],'group_id':row['group_id'],'margin_A':float(a['delta'][0]),
                'margin_B':float(b['delta'][0]),'margin_empty':float(empty['delta'][0])}
        outputs=[]
        for label,mask in masks.items():
            for kind in ('remove','keep','noise','denoise'):
                r=forward(mask,kind)
                outputs.append({**common,**label._asdict(),'intervention':kind,'margin':float(r['delta'][0]),
                    **route_change(b['route'] if kind=='denoise' else a['route'],r['route'])})
    return outputs


def summarize(records):
    groups={}
    for r in records:
        key=(r['ranking'],r['seed'],r['budget'],r['intervention'])
        groups.setdefault(key,[]).append(r)
    out=[]
    for key,rows in groups.items():
        mean=lambda field:float(np.mean([r[field] for r in rows]))
        kind=key[-1];m=mean('margin');a=mean('margin_A');b=mean('margin_B');empty=mean('margin_empty')
        # Ratio of means, never mean of per-row ratios; do not clip to [0,1].
        numerator,denominator=(m-b,a-b) if kind=='denoise' else (a-m,a-b) if kind=='noise' else (m-empty,a-empty)
        out.append(dict(zip(('ranking','seed','budget','intervention'),key),n=len(rows),
            margin=m,margin_A=a,margin_B=b,margin_empty=empty,delta_from_A=m-a,
            normalized=numerator/denominator if abs(denominator)>1e-6 else None,
            denominator=denominator,small_denominator=abs(denominator)<=1e-6,
            accuracy_A=float(np.mean([r['margin_A']>0 for r in rows])),
            accuracy_after=float(np.mean([r['margin']>0 for r in rows])),
            mean_changed_routes=mean('topk_changed_layer_positions')))
    return out
