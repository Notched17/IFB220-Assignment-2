#!/bin/bash
# Final verification on the final code/config: held-out x3 (where applicable), frozen x1.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY redteam_output/run_redteam.py --run 1 --prompts prompts_heldout.json --out _heldout_final --max-calls 500 > redteam_output/heldout_final_run1.log 2>&1 || exit 1
IDS=$($PY -c "
import json
R=[json.loads(l) for l in open('redteam_output/results_heldout_final.jsonl')]
print(','.join(sorted({r['id'] for r in R if r['action'] in ('answered','refused_output') or r['stopped_by']=='provider_filter'})))")
for r in 2 3; do
  $PY redteam_output/run_redteam.py --run $r --prompts prompts_heldout.json --out _heldout_final --ids "$IDS" --max-calls 500 > redteam_output/heldout_final_run$r.log 2>&1 || exit 1
done
$PY redteam_output/run_redteam.py --run 1 --prompts prompts_frozen.json --out _frozen_final --max-calls 900 > redteam_output/frozen_final_run1.log 2>&1
tail -n1 redteam_output/heldout_final_run*.log redteam_output/frozen_final_run1.log
