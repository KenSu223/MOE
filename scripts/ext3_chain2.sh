#!/usr/bin/env bash
# ext3 chain 2: extra position-0 substitutions (attached '.', ':', 'of', lone space, <unk>) after the pass-B result that
# '▁.' forms no sink while ',' and '\n' do. One batched Mixtral pass.
cd /home/ubuntu/MOE
bash scripts/gpu_queue.sh ext3-mixtral-passD -- python scripts/ext3_run_variants.py mixtral --variants dot2,colon,of,space,unk
echo "$(date -u +%FT%TZ) chain2 done"
