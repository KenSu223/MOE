"""CPU tests of real tiny HF OLMoE and the complete SVA handoff path."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moetrace import relp_sva as S
from moetrace.ext14_circuit_relp import FModel

torch.set_num_threads(1)

@pytest.fixture
def tiny(tmp_path):
    from transformers import OlmoeConfig,OlmoeForCausalLM
    torch.manual_seed(19)
    c=OlmoeConfig(vocab_size=32,hidden_size=16,intermediate_size=12,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=64,bos_token_id=0,eos_token_id=1,pad_token_id=0)
    model=OlmoeForCausalLM(c)
    d=tmp_path/'model';model.save_pretrained(d)
    fm=FModel(str(d),'cpu',torch.float32)
    row={'row_id':'tiny:1','group_id':'group1','input_ids':[2,3,4],'reference_ids':[2,5,4],
         'target_id':6,'foil_id':7,'special_tokens_mask':[0,0,0]}
    return fm,row,d

def test_native_forward_and_actual_output_patches(tiny):
    fm,row,_=tiny
    report=S.verify(fm,[row],1e-5,1e-4,1e-4)
    assert report['passed'],[c for c in report['checks'] if not c['passed']]

def test_half_rule_and_frozen_normalization_coefficients(tiny):
    fm,_,_=tiny
    g=torch.tensor([1.,-2.,.4],requires_grad=True)
    u=torch.tensor([.3,.7,-.2],requires_grad=True)
    activation=fm._act(g,u,True)
    dg,du=torch.autograd.grad(activation.sum(),(g,u))
    torch.testing.assert_close(dg,.5*torch.sigmoid(g)*u)
    torch.testing.assert_close(du,.5*torch.nn.functional.silu(g))
    torch.testing.assert_close(g*dg+u*du,activation)
    x=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
    y=fm.rms(x,torch.ones(4),True)
    dx,=torch.autograd.grad(y.sum(),x)
    torch.testing.assert_close(dx,torch.rsqrt(x.detach().square().mean(-1,keepdim=True)+fm.eps).expand_as(x))

def test_verifier_detects_wrong_projection(tiny):
    fm,row,_=tiny
    fm.layers[0]['down']=fm.layers[0]['down'].clone()+.4
    report=S.verify(fm,[row],1e-5,1e-4,1e-4)
    assert not report['passed']
    assert any('/HF/' in c['name'] and not c['passed'] for c in report['checks'])

def test_full_signed_mean_and_expert_cancellation():
    first=torch.tensor([[[[10.,9.,0.]]]])
    second=torch.tensor([[[[0.,9.,10.]]]])
    mean=(first+second)/2
    ranked=S.rankings(mean,torch.ones_like(mean,dtype=torch.bool),1)
    assert ranked['neuron'][0]['coordinate'][-1]==1  # never individual top-1
    x=torch.tensor([[[[5.,-5.],[3.,0.]]]])
    ranked=S.rankings(x,torch.ones_like(x,dtype=torch.bool))
    assert ranked['expert_position'][0]['coordinate']==[0,0,1]
    assert ranked['expert_position'][1]['score']==0

def test_random_control_preserves_layer_position_and_support():
    eligible=torch.tensor([[[True,True,False],[True,False,True]],[[True,True,True],[False,True,True]]])
    chosen=torch.zeros_like(eligible);chosen[0,0,0]=True;chosen[1,1,1]=True
    matched=S.random_matched(chosen,eligible,torch.Generator().manual_seed(101))
    assert torch.equal(chosen.sum(-1),matched.sum(-1))
    assert not (matched&~eligible).any()

def test_paired_endpoint_and_expert_identity(tiny):
    fm,row,_=tiny
    scores,support,_=S.score_pair(fm,row,'paired','all')
    a=fm.forward(S.tensor(fm,row),[6],[7],record=True)
    b=fm.forward(S.tensor(fm,row,True),[6],[7],record=True)
    _,betas=fm.node_grads(S.tensor(fm,row),[6],[7],True)
    for l in range(fm.L):
        expected=(a['q'][l]-b['q'][l])*betas[l]
        torch.testing.assert_close(scores[l],expected[0])
        out=a['out'][l].clone().requires_grad_()
        mask=torch.ones(1,3,dtype=torch.bool)
        result=fm.forward(S.tensor(fm,row),[6],[7],relp=True,out_patch={l:(mask,out)})
        lam,=torch.autograd.grad(result['delta'].sum(),out)
        dr=torch.einsum('btei,ehi->bteh',a['q'][l]-b['q'][l],fm.layers[l]['down'])
        direct=(dr*lam[:,:,None,:]).sum(-1)[0]
        torch.testing.assert_close(scores[l].sum(-1),direct,atol=1e-7,rtol=1e-4)
    assert support.any()

def test_data_profiles_are_locked_and_grouped_split_is_disjoint():
    for name in ('paper','grouped','smoke'):
        root=ROOT/'data/relp_sva'/f'olmoe_{name}_v1'
        meta=json.loads((root/'manifest.json').read_text())
        for task,entry in meta['tasks'].items():
            split_groups={}
            for split,expected in entry['files_sha256'].items():
                path=root/task/f'{split}.jsonl'
                assert S.sha(path)==expected
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                split_groups[split]={r['group_id'] for r in rows}
                for row in rows:
                    assert len(row['input_ids'])==len(row['reference_ids'])==len(entry['signature'])
                    assert row['token_word_map']==entry['signature']
                    assert row['target_id']!=row['foil_id']
                if name=='paper':assert len(rows)==(300 if split=='discovery' else 40)
            if name=='grouped':
                keys=list(split_groups)
                for i,a in enumerate(keys):
                    for b in keys[i+1:]:assert not split_groups[a]&split_groups[b]

def test_summary_uses_ratio_of_means_and_never_clips():
    rows=[dict(ranking='relp',seed=0,budget=1,intervention='denoise',margin=12.,margin_A=10.,margin_B=0.,margin_empty=0.,topk_changed_layer_positions=0),
          dict(ranking='relp',seed=0,budget=1,intervention='denoise',margin=2.,margin_A=1.,margin_B=0.,margin_empty=0.,topk_changed_layer_positions=0)]
    out=S.summarize(rows)[0]
    assert out['normalized']==pytest.approx(14/11)

def test_calibration_counts_routing_even_for_zero_activations(tiny):
    fm,row,_=tiny
    for layer in fm.layers:layer['gate_up'].zero_()
    mean,counts=S.calibrate_q(fm,[row])
    assert mean.abs().max()==0
    assert int(counts.sum())==fm.L*len(row['input_ids'])*fm.k

def test_conditional_mean_keep_all_is_original_forward(tiny):
    from collections import namedtuple
    fm,row,_=tiny
    mean,counts=S.calibrate_q(fm,[row])
    Label=namedtuple('Label','ranking seed budget')
    mask=torch.ones(fm.L,len(row['input_ids']),fm.E,fm.I,dtype=torch.bool)
    records=S.evaluate_case(fm,row,{Label('relp',0,int(mask.sum())):mask},'mean_q_routed','all',mean)
    keep=next(r for r in records if r['intervention']=='keep')
    remove=next(r for r in records if r['intervention']=='remove')
    assert keep['margin']==pytest.approx(keep['margin_A'],abs=1e-7)
    assert remove['margin']==pytest.approx(remove['margin_empty'],abs=1e-7)

def test_cli_verify_score_evaluate_and_stale_gate(tiny,tmp_path):
    fm,row,model_dir=tiny
    root=tmp_path/'manifest';task=root/'simple';task.mkdir(parents=True)
    hashes={}
    for split in ('calibration','discovery','validation','test'):
        path=task/f'{split}.jsonl';path.write_text(json.dumps(row)+'\n');hashes[split]=S.sha(path)
    manifest={'model':{'revision':'synthetic'},'source_revision':'synthetic',
        'tokenizer_sha256':{'config.json':S.sha(model_dir/'config.json')},
        'tasks':{'simple':{'files_sha256':hashes}}}
    S.write_json(root/'manifest.json',manifest)
    script=ROOT/'scripts/relp_sva_run.py'
    common=['--model-dir',str(model_dir),'--manifest-dir',str(root),'--task','simple','--device','cpu','--dtype','float32']
    def run(command,extra):
        result=subprocess.run([sys.executable,str(script),command]+common+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    ver=tmp_path/'verify'
    run('verify',['--output-dir',str(ver),'--verify-cases','1'])
    score=tmp_path/'score'
    run('score',['--output-dir',str(score),'--verification',str(ver/'verification.json'),
         '--neuron-budgets','1','2','--expert-budgets','1','--random-seeds','101'])
    assert (score/'expert_position_ranking.json').exists()
    for unit in ('neuron','expert_position','expert_all_positions'):
        dest=tmp_path/('evaluate_'+unit)
        run('evaluate',['--output-dir',str(dest),'--verification',str(ver/'verification.json'),
            '--score-dir',str(score),'--unit',unit])
        assert json.loads((dest/'run_meta.json').read_text())['status']=='complete'
        assert len(json.loads((dest/'summary.json').read_text()))>0
    # A different declared dtype must reject the certificate, without relying on a filename.
    data=json.loads((ver/'verification.json').read_text());data['identity']['dtype']='stale-dtype'
    S.write_json(ver/'bad.json',data)
    result=subprocess.run([sys.executable,str(script),'score']+common+[
        '--output-dir',str(tmp_path/'rejected'),'--verification',str(ver/'bad.json')],capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'rejected').exists()
