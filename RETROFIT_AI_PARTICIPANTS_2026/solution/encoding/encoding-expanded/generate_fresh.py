"""Reserve the new seeds explicitly, then generate only after the final pipeline freeze."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'generator'))
import synthetic


def main():
    p=argparse.ArgumentParser();p.add_argument('--reserve-only',action='store_true');args=p.parse_args()
    plan=json.loads((ROOT/'fresh_test_plan.json').read_text())
    seeds=tuple(plan['new_seeds'])
    if set(seeds)&set(plan['previously_used_seeds']):raise ValueError('Seeds were previously used')
    # The copied generator supports custom seeds; ensure its reservation file records the actual override too.
    synthetic.ROLE_SEEDS['holdout']=seeds
    output=ROOT/'synthetic_data';synthetic.reserve_seeds(output)
    if args.reserve_only:
        print('Reserved',seeds);return
    frozen=ROOT/'results/frozen.json'
    if json.loads(frozen.read_text())['test_seeds']!=list(seeds):raise ValueError('Frozen seeds differ')
    synthetic.generate_role(ROOT.parents[1]/'RETROFIT_AI_PARTICIPANTS_2026',output,'holdout',seeds,3,frozen)


if __name__=='__main__':main()
