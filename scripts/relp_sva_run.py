#!/usr/bin/env python3
"""Verify, discover, and evaluate fixed SVA neuron/expert lists. No training of model weights."""
from __future__ import annotations
import argparse
from collections import namedtuple
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','score','evaluate'))
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--manifest-dir',type=Path,required=True)
    p.add_argument('--task',choices=('simple','rc','within_rc','nounpp'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=('bfloat16','float32'),default='bfloat16')
    p.add_argument('--verification',type=Path,help='Passed verify/verification.json from this installation')
    p.add_argument('--score-dir',type=Path)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--mode',choices=('paired','unpaired'),default='paired')
    p.add_argument('--pool',choices=('non_special','all','final'),default='non_special')
    p.add_argument('--unit',choices=('neuron','expert_position','expert_all_positions'),default='neuron')
    p.add_argument('--baseline',choices=('zero','mean_q_routed'),default='zero',help='Lock at score time')
    p.add_argument('--neuron-budgets',nargs='+',type=int,default=[1,5,10,50,100,200,500,1000])
    p.add_argument('--expert-budgets',nargs='+',type=int,default=[1,2,5,10,20,50])
    p.add_argument('--random-seeds',nargs='+',type=int,default=[101,102,103])
    p.add_argument('--verify-cases',type=int,default=4)
    p.add_argument('--atol',type=float,default=1e-4)
    p.add_argument('--rtol',type=float,default=1e-3)
    p.add_argument('--max-relative',type=float,default=1e-3)
    a=p.parse_args()
    if a.output_dir.exists():p.error('Output directory exists; do not overwrite runs')
    if any(n<=0 for n in a.neuron_budgets+a.expert_budgets) or a.verify_cases<1:
        p.error('Budgets and verify-cases must be positive')
    import torch
    import pandas as pd
    from moetrace import relp_sva as S
    from moetrace.ext14_circuit_relp import FModel
    from scripts.relp_score_pair import check_loaded_layout
    meta,calibration=S.read_split(a.manifest_dir,a.task,'calibration')
    S.check_tokenizer_files(a.model_dir,meta)
    # Reuse architecture/config guards without interpreting IDs from other models.
    config=json.loads((a.model_dir/'config.json').read_text())
    if config.get('model_type') not in ('olmoe','mistral') or config.get('hidden_act','silu')!='silu':
        p.error('Unsupported model/activation')
    if config.get('clip_qkv') is not None or any(config.get(k) for k in ('attention_bias','mlp_bias','num_shared_experts','shared_expert_intermediate_size')):
        p.error('Unsupported clipping/bias/shared-expert configuration')
    length=len(calibration[0]['input_ids'])
    for k in ('sliding_window','max_position_embeddings'):
        bound=config.get(k)
        if isinstance(bound,int) and bound>0 and length>bound:p.error('Sequence exceeds '+k)
    fm=FModel(str(a.model_dir.resolve()),device=a.device,dtype=getattr(torch,a.dtype))
    check_loaded_layout(fm)
    ident=S.identity(fm,a.model_dir)
    common={'utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'task':a.task,
            'manifest_sha256':S.sha(a.manifest_dir/'manifest.json'),'identity':ident,
            'model_revision':meta['model']['revision'],'source_revision':meta['source_revision'],
            'script_sha256':S.sha(__file__),'status':'running'}
    if a.command!='verify':
        if a.verification is None:p.error('--verification is required; run native checks first')
        v=json.loads(a.verification.read_text())
        if not v.get('passed') or v['identity']!=ident or v['task']!=a.task or v['manifest_sha256']!=common['manifest_sha256']:
            p.error('Verification is failed, stale, or for another model/task/manifest/runtime')
        common['verification_sha256']=S.sha(a.verification)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    S.write_json(a.output_dir/'run_meta.json',common)
    if a.command=='verify':
        report=S.verify(fm,calibration[:a.verify_cases],a.atol,a.rtol,a.max_relative)
        report.update(common);report['passed']=all(c['passed'] for c in report['checks'])
        report['status']='complete' if report['passed'] else 'failed'
        S.write_json(a.output_dir/'verification.json',report)
        S.write_json(a.output_dir/'run_meta.json',{**common,'status':'complete' if report['passed'] else 'failed'})
        print(json.dumps({'passed':report['passed'],'failed_checks':[c for c in report['checks'] if not c['passed']]}))
        if not report['passed']:raise SystemExit(1)
        return
    if a.command=='score':
        _,rows=S.read_split(a.manifest_dir,a.task,'discovery')
        total=support=None;cases=[]
        stream=(a.output_dir/'cases.jsonl').open('x')
        try:
            for i,row in enumerate(rows):
                score,active,record=S.score_pair(fm,row,a.mode,a.pool)
                if total is None:total=torch.zeros_like(score,dtype=torch.float64);support=torch.zeros_like(active)
                total+=score.double();support|=active
                cases.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
                if (i+1)%10==0:print(f'scored {i+1}/{len(rows)}',flush=True)
        finally:stream.close()
        mean=total/len(rows)
        payload={'mean':mean,'support':support,'n':len(rows)}
        if a.baseline=='mean_q_routed':
            payload['qmean'],payload['calibration_counts']=S.calibrate_q(fm,calibration)
        torch.save(payload,a.output_dir/'aggregate.pt')
        lists=S.rankings(mean,support,max(a.neuron_budgets+[1000]))
        for unit,values in lists.items():
            S.write_json(a.output_dir/f'{unit}_ranking.json',values)
        pd.DataFrame(cases).to_parquet(a.output_dir/'cases.parquet',index=False)
        plan={'mode':a.mode,'endpoint':'A','pool':a.pool,'baseline':a.baseline,
              'neuron_budgets':sorted(set(a.neuron_budgets)),'expert_budgets':sorted(set(a.expert_budgets)),
              'random_seeds':a.random_seeds,'ranking':'abs(mean(signed score))',
              'expert_aggregation':'sum over all neurons before ranking',
              'mean_baseline':'E[q | expert selected], per layer-position-expert on calibration A only; applied to live-routed experts',
              'discovery_sha256':meta['tasks'][a.task]['files_sha256']['discovery'],
              'n_discovery':len(rows),'aggregate_sha256':S.sha(a.output_dir/'aggregate.pt')}
        S.write_json(a.output_dir/'selection_plan.json',plan)
        S.write_json(a.output_dir/'run_meta.json',{**common,'status':'complete','selection_plan_sha256':S.sha(a.output_dir/'selection_plan.json')})
        print('Saved complete signed means and independent neuron/expert rankings.',flush=True)
        return
    if a.score_dir is None:p.error('--score-dir required for evaluation')
    scoremeta=json.loads((a.score_dir/'run_meta.json').read_text())
    for key in ('identity','task','manifest_sha256'):
        if scoremeta[key]!=common[key]:raise ValueError('Score run mismatch: '+key)
    if scoremeta['status']!='complete':raise ValueError('Incomplete discovery run')
    plan=json.loads((a.score_dir/'selection_plan.json').read_text())
    if S.sha(a.score_dir/'selection_plan.json')!=scoremeta['selection_plan_sha256'] or S.sha(a.score_dir/'aggregate.pt')!=plan['aggregate_sha256']:
        raise ValueError('Discovery output changed after selection was locked')
    payload=torch.load(a.score_dir/'aggregate.pt',map_location='cpu',weights_only=True)
    mean,support=payload['mean'],payload['support']
    if a.unit=='neuron':values,eligible=mean,support
    elif a.unit=='expert_position':values,eligible=mean.sum(-1),support.any(-1)
    else:values,eligible=mean.sum(-1).sum(1),support.any(-1).any(1)
    budgets=plan['neuron_budgets' if a.unit=='neuron' else 'expert_budgets']
    Label=namedtuple('Label','ranking seed budget')
    masks={}
    order=S.rank_indices(values,eligible,max(budgets))
    if len(order)<max(budgets):raise ValueError('Budget exceeds eligible units; revise discovery plan on development data')
    for n in budgets:
        chosen=torch.zeros_like(eligible);chosen.reshape(-1)[order[:n]]=True
        def make(x):
            if a.unit=='neuron':return x
            if a.unit=='expert_position':return x[...,None].expand_as(mean)
            return x[:,None,:,None].expand_as(mean)
        masks[Label('relp',0,n)]=make(chosen)
        for seed in plan['random_seeds']:
            # Fixed across cases; match layer/position and discovery-active support.
            if a.unit=='expert_all_positions':
                random=S.random_matched(chosen[:,None,:],eligible[:,None,:],torch.Generator().manual_seed(seed+n))[:,0,:]
            else:random=S.random_matched(chosen,eligible,torch.Generator().manual_seed(seed+n))
            masks[Label('random',seed,n)]=make(random)
    _,rows=S.read_split(a.manifest_dir,a.task,a.split)
    records=[]
    with (a.output_dir/'interventions.jsonl').open('x') as stream:
        for i,row in enumerate(rows):
            result=S.evaluate_case(fm,row,masks,plan['baseline'],plan['pool'],payload.get('qmean'))
            records.extend(result)
            for record in result:stream.write(json.dumps(record)+'\n')
            stream.flush();print(f'evaluated {i+1}/{len(rows)}',flush=True)
    pd.DataFrame(records).to_parquet(a.output_dir/'interventions.parquet',index=False)
    S.write_json(a.output_dir/'summary.json',S.summarize(records))
    S.write_json(a.output_dir/'run_meta.json',{**common,'status':'complete','split':a.split,'unit':a.unit,
        'selection_plan':plan,'n':len(rows),'evaluation':'original nonlinear FModel after native acceptance; downstream natural routing; no rerouting/renormalization at patched q',
        'normalization':'ratio of means; keep=faithfulness, remove=remaining(completeness), noise=lost paired difference, denoise=recovery'})


if __name__=='__main__':main()
