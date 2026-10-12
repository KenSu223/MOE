#!/usr/bin/env bash
# ext13 p5-data: resume the base chain after the chain process was stopped while a job ($1 = job name, $2 = run dir,
# $3 = stage) was still running: wait for that job's END line, mark the stage done on rc=0, then restart the base chain.
cd "$(dirname "$0")/.."
name=$1; run=$2; st=$3; shift 3
until grep -q "END   $name rc=" "logs/$name.log" 2>/dev/null; do sleep 15; done
rc=$(grep "END   $name rc=" "logs/$name.log" | tail -1 | sed -E 's/.*rc=([0-9]+).*/\1/')
echo "$(date -u +%FT%TZ) $name rc=$rc (finished after chain restart)" >> logs/ext13_p5_chain.log
[ "$rc" = "0" ] && touch "results/$run/.done_$st"
exec bash scripts/ext13_p5_chain_base.sh "$@"
