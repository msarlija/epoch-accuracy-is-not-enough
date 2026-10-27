"""Build an "ensemble run" directory per arm so the paper's analysis scripts can
score the ten-fold ensemble unchanged (as prefix ``ens``, one fold: --prefix ens --folds 1).

For every external recording of the capped set, the softmax of the ten fold
models is averaged (stage_b and stage_a separately) and the hypnogram is its
argmax. Written as runs/ens_<arm>_seed0_fold0of10/<stage>/predictions_external_v2/
<rec>.npz plus <stage>/external_v2/per_recording_metrics.csv and the other
tables that run_experiment.write_metrics produces. Internal test recordings
are scored by one fold model each, so no internal ensemble exists.

    python scripts/make_ensemble_run.py [--arms full zero none] [--prefix final]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import load_config  # noqa: E402
from scripts.run_experiment import write_metrics  # noqa: E402

RUNS = os.path.expanduser(os.environ.get('SLEEP_DATA_ROOT', '~/data/sleep_staging') + '/runs')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', default=['full', 'zero'])
    ap.add_argument('--prefix', default='final')
    a = ap.parse_args()
    for arm in a.arms:
        src0 = f'{RUNS}/{a.prefix}_{arm}_seed0_fold0of10'
        eval_cfg = load_config(f'{src0}/resolved_config.yaml').get('eval', {})
        meta = pd.read_csv(f'{src0}/stage_b/external_v2/per_recording_metrics.csv')
        out = f'{RUNS}/ens_{arm}_seed0_fold0of10'
        for st in ('stage_b', 'stage_a'):
            pdir = f'{out}/{st}/predictions_external_v2'
            os.makedirs(pdir, exist_ok=True)
            recs = []
            for _, r in meta.iterrows():
                zs = [np.load(f'{RUNS}/{a.prefix}_{arm}_seed0_fold{f}of10/{st}/predictions_external_v2/{r.rec_id}.npz')
                      for f in range(10)]
                n = min(len(z['proba']) for z in zs)
                proba = np.mean([z['proba'][:n] for z in zs], 0).astype(np.float32)
                y = zs[0]['y_true'].astype(np.int64)[:n]
                pred = np.where(y == -1, -1, proba.argmax(1))
                np.savez_compressed(f'{pdir}/{r.rec_id}.npz', y_true=y.astype(np.int8), y_pred=pred.astype(np.int8),
                                    proba=proba)
                recs.append(dict(rec_id=r.rec_id, cohort=r.cohort, subject_id=str(r.subject_id),
                                 montage_family=r.get('montage_family', ''), n_channels=int(r.get('n_channels', 0)),
                                 y_true=y, y_pred=pred, proba=proba))
            T = write_metrics(recs, f'{out}/{st}/external_v2', eval_cfg)
            print(arm, st, len(T), 'recordings, kappa', round(T.kappa.mean(), 4), flush=True)
        for f in ('resolved_config.yaml', 'split_manifest.json'):
            if not os.path.exists(f'{out}/{f}'):
                os.symlink(f'{src0}/{f}', f'{out}/{f}')
        if not os.path.exists(f'{out}/stage_b/predictions_external'):
            os.symlink('predictions_external_v2', f'{out}/stage_b/predictions_external')
        print('wrote', out)


if __name__ == '__main__':
    main()
