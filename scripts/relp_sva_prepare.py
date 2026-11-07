#!/usr/bin/env python3
"""Fetch pinned SVA data/tokenizer, then build auditable model-specific manifests (no weights)."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/relp_sva'
TASKS = ('simple', 'rc', 'within_rc', 'nounpp')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def download(url, path, size=None, blob=None):
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(url, timeout=120).read()
    if size is not None and len(data) != size:
        raise ValueError(f'Wrong size: {path}')
    if blob and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() != blob:
        raise ValueError(f'Wrong Git blob hash: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def fetch(root):
    lock = json.loads((root/'sources.lock.json').read_text())
    def one(f):
        url = f"https://raw.githubusercontent.com/{lock['repository']}/{lock['revision']}/{f['path']}"
        sha = download(url, root/'raw'/Path(f['path']).name, f['size'], f['git_blob_sha1'])
        return {'path':f['path'], 'sha256':sha, 'git_blob_sha1':f['git_blob_sha1']}
    with ThreadPoolExecutor(max_workers=4) as ex:
        files = list(ex.map(one, lock['files']))
    model = lock['model']
    tok = {}
    for name in model['tokenizer_files']:
        tok[name] = download(f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}", root/'tokenizer'/name)
        if name in model.get('tokenizer_sha256',{}) and tok[name]!=model['tokenizer_sha256'][name]:
            raise ValueError(f'Pinned tokenizer hash mismatch: {name}')
    dump(root/'download_receipt.json', {'source_revision':lock['revision'], 'files':files,
         'model':model, 'tokenizer_sha256':tok, 'model_weights_downloaded':False})

def token_word_map(text, offsets, special):
    words = list(re.finditer(r'\S+', text))
    result = []
    for (a,b), sp in zip(offsets, special):
        if sp:
            result.append(-1)
            continue
        hits = [i for i,w in enumerate(words) if max(a,w.start()) < min(b,w.end())]
        if len(hits) != 1:
            raise ValueError('ambiguous_token_word_alignment')
        result.append(hits[0])
    return result

def encode_rows(rows, tok, task):
    # Batch unique strings once. rc contains >100k source rows with many duplicates.
    texts = sorted({s for row in rows for s in [row['clean_prefix'], row['patch_prefix']]
                    + [row[p]+row[a] for p in ('clean_prefix','patch_prefix') for a in ('clean_answer','patch_answer')]})
    enc = tok(texts, add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
    cache = {s:(ids,off,sp) for s,ids,off,sp in zip(texts,enc['input_ids'],enc['offset_mapping'],enc['special_tokens_mask'])}
    accepted, rejected = [], Counter()
    for row in rows:
        try:
            a,b = row['clean_prefix'], row['patch_prefix']
            ai,ao,asp = cache[a]; bi,bo,bsp = cache[b]
            if len(ai) != len(bi): raise ValueError('pair_length_mismatch')
            if ai == bi: raise ValueError('identical_input_pair')
            am = token_word_map(a,ao,asp); bm = token_word_map(b,bo,bsp)
            if am != bm or len(a.split()) != len(b.split()): raise ValueError('pair_word_alignment_mismatch')
            if len(a.split()) != {'simple':2,'rc':6,'within_rc':5,'nounpp':5}[task]:
                raise ValueError('unexpected_template_length')
            candidates = []
            for answer in (row['clean_answer'],row['patch_answer']):
                ids = []
                for prompt,prefix in ((a,ai),(b,bi)):
                    combined = cache[prompt+answer][0]
                    if combined[:len(prefix)] != prefix: raise ValueError('continuation_changes_prefix')
                    suffix = combined[len(prefix):]
                    if len(suffix) != 1: raise ValueError('answer_not_single_token')
                    ids.append(suffix[0])
                if ids[0] != ids[1]: raise ValueError('candidate_id_differs_across_pair')
                candidates.append(ids[0])
            if candidates[0] == candidates[1]: raise ValueError('same_target_foil')
            group = digest(sorted((a,b)))
            accepted.append({**row, 'input_ids':ai, 'reference_ids':bi,
                'target_id':candidates[0], 'foil_id':candidates[1], 'token_word_map':am,
                'tokens_input':tok.convert_ids_to_tokens(ai), 'tokens_reference':tok.convert_ids_to_tokens(bi),
                'special_tokens_mask':asp, 'changed_positions':[i for i,(u,v) in enumerate(zip(ai,bi)) if u!=v],
                'subject_word':4 if task=='within_rc' else 1, 'group_id':group,
                'content_id':digest([a,b,row['clean_answer'],row['patch_answer']])})
        except ValueError as error:
            rejected[str(error)] += 1
    return accepted, dict(rejected)

def overlap(splits):
    out = {}
    for name,rows in splits.items():
        out[name] = {'rows':len(rows),'unique_content':len({r['content_id'] for r in rows}),
                     'unique_groups':len({r['group_id'] for r in rows}), 'case_counts':dict(Counter(r['case'] for r in rows))}
    pairs = {}
    keys = list(splits)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            pairs[a+'|'+b] = {field:len({r[field] for r in splits[a]} & {r[field] for r in splits[b]})
                              for field in ('content_id','group_id')}
    return {'splits':out,'overlaps':pairs}

def select_splits(train, test, profile, seed):
    rng = random.Random(seed)
    train, test = list(train), list(test)
    rng.shuffle(train)
    random.Random(seed).shuffle(test)
    if profile in ('paper','smoke'):
        counts = (300,40,40,40) if profile=='paper' else (20,8,8,0)
        nd,nc,nv,nt = counts
        if len(train)<nd+nc+nv or len(test)<nt:
            raise ValueError(f'Insufficient eligible rows: train={len(train)} test={len(test)} requested={counts}')
        return dict(discovery=train[:nd],calibration=train[nd:nd+nc],validation=train[nd+nc:nd+nc+nv],test=test[:nt])
    # Strict supplementary experiment: group both orientations and all verb answers together.
    # Existing source file split is intentionally replaced; duplicate content is removed.
    unique = {r['content_id']:r for r in reversed(train+test)}
    groups = sorted({r['group_id'] for r in unique.values()})
    rng.shuffle(groups)
    n = len(groups)
    if n<8: raise ValueError('Too few independent prefix groups for grouped split')
    cuts = (int(.60*n),int(.75*n),int(.85*n))
    partitions = [groups[:cuts[0]],groups[cuts[0]:cuts[1]],groups[cuts[1]:cuts[2]],groups[cuts[2]:]]
    out = {}
    for name,ids,cap in zip(('discovery','calibration','validation','test'),partitions,(300,40,40,40)):
        ids = set(ids)
        pool = sorted((r for r in unique.values() if r['group_id'] in ids),key=lambda r:r['content_id'])
        rng.shuffle(pool)
        out[name] = pool[:cap]
    return out

def prepare(args):
    from transformers import AutoTokenizer
    import transformers, tokenizers
    root, output = args.data_root, args.output_dir
    if output.exists(): raise ValueError('Output already exists; choose a new directory')
    lock=json.loads((root/'sources.lock.json').read_text())
    tokdir=args.tokenizer_dir or root/'tokenizer'
    for name,expected in lock['model'].get('tokenizer_sha256',{}).items():
        if hashlib.sha256((tokdir/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Tokenizer/config is not the pinned OLMoE revision: {name}')
    tok=AutoTokenizer.from_pretrained(tokdir, local_files_only=True, use_fast=True)
    if not tok.is_fast: raise ValueError('Fast tokenizer required for offset alignment')
    tok_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tokdir.glob('*'))
              if p.is_file() and ('token' in p.name or p.name=='config.json')}
    task_outputs={}
    for task in args.tasks:
        raw={}
        for split in ('train','test'):
            name=f'{task}_{split}.json'
            path=root/'raw'/name
            spec=next(f for f in lock['files'] if Path(f['path']).name==name)
            data=path.read_bytes()
            if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_sha1']:
                raise ValueError(f'Source data hash mismatch: {path}')
            raw[split]=[{**json.loads(line),'row_id':f'{task}:{split}:{i+1}', 'source_split':split}
                        for i,line in enumerate(data.decode().splitlines()) if line.strip()]
        encoded,rejects=encode_rows(raw['train']+raw['test'],tok,task)
        train=[r for r in encoded if r['source_split']=='train']
        # Freeze a token-to-word signature from training only; apply it unchanged to every split.
        counts=Counter(tuple(r['token_word_map']) for r in train)
        if not counts: raise ValueError(f'No aligned training rows: {task}')
        signature=min(counts, key=lambda s:(len(s),-counts[s],s))
        eligible={s:[r for r in encoded if r['source_split']==s and tuple(r['token_word_map'])==signature]
                  for s in ('train','test')}
        splits=select_splits(eligible['train'],eligible['test'],args.profile,args.seed)
        task_outputs[task]={'task':task,'signature':list(signature),'splits':splits,
            'funnel':{'raw':{s:len(v) for s,v in raw.items()},'rejected_checks':rejects,
                      'eligible_signature':{s:len(v) for s,v in eligible.items()},
                      'other_signature':len(encoded)-sum(map(len,eligible.values()))},'audit':overlap(splits)}
    output.mkdir(parents=True,exist_ok=False)
    meta={'schema':1,'profile':args.profile,'seed':args.seed,'source_repository':lock['repository'],
          'source_revision':lock['revision'],'model':lock['model'],'tokenizer_sha256':tok_hash,
          'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,
          'tokenization':'raw prefix, add_special_tokens=True, no chat template, no padding',
          'alignment':'same token-to-whitespace-word map, shortest training signature, expert IDs fixed',
          'selection_uses_model_outputs':False,'paper_exact_row_ids_claimed':False,
          'profile_note':'paper preserves duplicate source rows; grouped replaces original file split and deduplicates content',
          'prepare_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'tasks':{}}
    for task,result in task_outputs.items():
        dest=output/task;dest.mkdir()
        fingerprints={}
        for name,rows in result.pop('splits').items():
            p=dest/f'{name}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            fingerprints[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        result['files_sha256']=fingerprints
        dump(dest/'selection.json',result)
        meta['tasks'][task]=result
        print(task,json.dumps(result['audit']['splits']),flush=True)
    dump(output/'manifest.json',meta)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('fetch','prepare'))
    p.add_argument('--data-root',type=Path,default=DEFAULT)
    p.add_argument('--tokenizer-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--profile',choices=('paper','grouped','smoke'),default='paper')
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    p.add_argument('--seed',type=int,default=12)
    a=p.parse_args()
    if a.command=='fetch':fetch(a.data_root)
    else:
        if a.output_dir is None:p.error('--output-dir is required')
        prepare(a)

if __name__=='__main__':main()
