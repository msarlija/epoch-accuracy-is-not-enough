# -*- coding: utf-8 -*-
"""Every adaptation number of the paper for the ten-fold ensemble (Tables 6 and 7,
Supplementary Tables S5, S13 and S14, Figure 4 and the text of Section 3.8). The adapted softmax of each fold model, dumped by
domain_adapt.py / da4_self_train.py --dump-proba (prior-corrected), is averaged
over the ten folds into one hypnogram per recording and compared with the
ensemble of the unadapted fold models. Strategies: DA-G (global), DA-MC
(multi-cluster), the single-anchor variant DA-A of Supplementary Table S13,
and the supervised reference DA-O (scripts/da4_self_train.py --dump-proba). A recording a strategy leaves unadapted
in a fold (too few anchor epochs) enters with its unadapted softmax. Statistics:
subject-clustered paired bootstrap over 301 subjects, one hypnogram each.

Writes <out>/da_ensemble_paper.txt (the report), analysis/da_ensemble_perrec.csv
(per recording and condition) and analysis/paper/da_runs/current_kcurve.csv,
the table behind Figure 4 (make_paper_figures.py kcurve).

    python scripts/ensemble_da_numbers.py [--nboot 10000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.epoch_metrics import epoch_metrics  # noqa: E402
from eval.macrostructure import record_macrostructure  # noqa: E402
from eval.stats import paired_bootstrap  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"
DA4 = f"{ROOT}/analysis/da4"
COH = ['ANPHY', 'DODH', 'DODO', 'DREAMS-P', 'DREAMS-S', 'MASS-SS1', 'WSC']
KS = ['1', '2', '5', '10', '15', 'all']
PROT = {'all': 'loso', '1': 'loso1', '2': 'loso2', '5': 'loso5', '10': 'loso10', '15': 'loso15',
        'trans': 'cohort'}
# DA-G global alignment, DA-A single anchor (N2 only, Supplementary Table S13),
# DA-MC multi-cluster; the tags are domain_adapt.py's run names
TAG = {'DA-G': 'da_da1_{p}_all_pre_n2_affine_c0.7_prior',
       'DA-A': 'da_da2_{p}_pred-conf_pre_n2_affine_c0.7_prior',
       'DA-MC': 'da_da2_{p}_pred-conf_pre_n2_multicluster-drop_c0.7_prior',
       'DA-MC expert': 'da_da2_{p}_true_pre_n2_multicluster-drop_c0.7_prior'}
NB = 10000


def metrics(y, h):
    k = y >= 0
    m = epoch_metrics(y[k], h[k])
    mac = record_macrostructure(y, np.where(k, h, -1))
    return dict(kappa=m['kappa'], accuracy=100 * m['accuracy'], f1_macro=100 * m['f1_macro'],
                rol_ae=mac['rol_ae_min'], w10=np.nan if np.isnan(mac['rol_ae_min']) else 100.0 * (mac['rol_ae_min'] <= 10),
                early=float(mac['rol_pred_min'] <= 15 and mac['rol_true_min'] > 15))


class Ens:
    def __init__(self, ids, subj, coh):
        self.ids, self.subj, self.coh = ids, subj, coh
        self.base = {rid: [np.load(f'{RUNS}/final_full_seed0_fold{f}of10/stage_b/predictions_external_v2/{rid}.npz')
                           for f in range(10)] for rid in ids}
        self.cache = {}

    def table(self, files, label):
        """files(rid, f) -> path of the adapted proba of fold f, or None for unadapted"""
        if label in self.cache:
            return self.cache[label]
        rows, missing = [], 0
        for rid in self.ids:
            P = []
            n = min(len(z['proba']) for z in self.base[rid])
            y = self.base[rid][0]['y_true'].astype(int)[:n]
            for f in range(10):
                fn = files(rid, f)
                if fn is not None and os.path.exists(fn):
                    P.append(np.load(fn)['proba'][:n])
                else:
                    P.append(self.base[rid][f]['proba'][:n]); missing += 1
            rows.append(dict(rec_id=rid, subject_id=self.subj[rid], cohort=self.coh[rid],
                             **metrics(y, np.mean(P, 0).argmax(1))))
        T = pd.DataFrame(rows)
        T.attrs['missing'] = missing
        self.cache[label] = T
        return T


def boot(A, B, col, nb=NB):
    m = A[['rec_id', 'cohort', 'subject_id', col]].merge(B[['rec_id', col]], on='rec_id', suffixes=('', '_b')).dropna()
    X = m[['rec_id', 'cohort', 'subject_id']].assign(v=m[col].values)
    Z = m[['rec_id']].assign(v=m[col + '_b'].values)
    r = paired_bootstrap(X, Z, 'v', n_boot=nb, seed=0)
    return dict(d=r['mean_diff'], lo=r['ci_low'], hi=r['ci_high'], p=r['p_value'], n=len(m))


def ratio(a, b):
    return f'{a / b:.1f}' if b else 'n/a'


def pct_of(a, b):
    return f'{100 * a / b:.0f}%' if b else 'n/a'


def fmt(r, nd=4):
    return f"{r['d']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}] p={r['p']:.4f} n={r['n']}"


def per_cohort(A, B, col='kappa', nb=2000):
    return {c: boot(A[A.cohort == c], B[B.cohort == c], col, nb) for c in COH}


def pc_str(d, nd=4):
    return '  '.join(f"{c}:{v['d']:+.{nd}f}" + (' n.s.' if v['p'] >= 0.05 else '') for c, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=ROOT)
    ap.add_argument('--nboot', type=int, default=10000)
    ap.add_argument('--dao-tag', default='da_o')
    a = ap.parse_args()
    global NB
    NB = a.nboot
    from config import load_config
    man = pd.read_csv(load_config('configs/final_full.yaml')['data_dir'] + '/manifest.csv')
    subj = dict(zip(man.rec_id, man.subject_id.astype(str)))
    coh = dict(zip(man.rec_id, man.cohort))
    ids = sorted(pd.read_csv(f'{RUNS}/final_full_seed0_fold0of10/stage_b/external_v2/per_recording_metrics.csv').rec_id)
    E = Ens(ids, subj, coh)
    out = []

    def say(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True); out.append(s)

    def da_files(strategy, prot):
        tag = TAG[strategy].format(p=prot)
        return lambda rid, f: f'{RUNS}/final_full_seed0_fold{f}of10/stage_b/proba_{tag}/{rid}.npz'

    def dao_files(K):
        sub = f'oracle_halves_k{K}_s2000_l1_lr0.0001_prior' if K != 'half' else 'oracle_halves_s2000_l1_lr0.0001_prior'
        return lambda rid, f: f'{DA4}/fold{f}_{a.dao_tag}/{sub}/proba/{rid}.npz'

    B0 = E.table(lambda rid, f: None, 'unadapted')
    say('ensemble of unadapted fold models: kappa', round(B0.kappa.mean(), 4), ' within-10', round(B0.w10.mean(), 1),
        ' false early', int(B0.early.sum()), 'of', int(B0.w10.notna().sum()))

    say('\n=== POOLED, leave-one-subject-out (Table 6; DA-A: Supplementary Table S13), ensemble')
    L = {s: E.table(da_files(s, 'loso'), s + ' loso') for s in ('DA-G', 'DA-A', 'DA-MC')}
    for s, T in L.items():
        pc = per_cohort(T, B0)
        worst = min(pc.items(), key=lambda kv: kv[1]['d'])
        say(f"{s}: kappa {fmt(boot(T, B0, 'kappa'))}  (unadapted in {T.attrs['missing']} recording-folds)")
        say(f"      acc pp {fmt(boot(T, B0, 'accuracy'), 2)}  F1 pp {fmt(boot(T, B0, 'f1_macro'), 2)}")
        say(f"      within-10 pp {fmt(boot(T, B0, 'w10'), 2)}  ROL AE min {fmt(boot(T, B0, 'rol_ae'), 2)}")
        say(f"      false early {int(B0.early.sum())} -> {int(T.early.sum())}  smallest cohort {worst[0]} {worst[1]['d']:+.4f}"
            + (' n.s.' if worst[1]['p'] >= 0.05 else ''))
    say('\n=== PER COHORT kappa (Table 7; DA-A: Supplementary Table S13), p<0.05 unless n.s.')
    for s, T in L.items():
        say(s, pc_str(per_cohort(T, B0)))
    say('\n=== DA-MC minus DA-A (ensembles, same recordings; Supplementary Table S13)')
    say('kappa pooled', fmt(boot(L['DA-MC'], L['DA-A'], 'kappa')))
    say('  per cohort', pc_str(per_cohort(L['DA-MC'], L['DA-A'])))
    say('within-10 pp', fmt(boot(L['DA-MC'], L['DA-A'], 'w10'), 2), ' ROL AE', fmt(boot(L['DA-MC'], L['DA-A'], 'rol_ae'), 2))
    w = per_cohort(L['DA-MC'], L['DA-A'], 'w10', 500)
    say('  within-10 ahead in', sum(v['d'] > 0 for v in w.values()), 'of 7 cohorts')
    say('\n=== Spearman of the DA-A lead against the unadapted ensemble kappa (7 cohorts)')
    lead = pd.Series({c: per_cohort(L['DA-A'], L['DA-MC'], 'kappa', 200)[c]['d'] for c in COH})
    uk = B0.groupby('cohort').kappa.mean()[COH]
    rho, p = stats.spearmanr(lead[COH], uk)
    say(f'  rho {rho:+.2f} p={p:.2f}')

    say('\n=== TRANSDUCTIVE against leave-one-subject-out, ensemble')
    for s in ('DA-G', 'DA-A', 'DA-MC'):
        T = E.table(da_files(s, 'cohort'), s + ' cohort')
        dt, dl = boot(T, B0, 'kappa')['d'], boot(L[s], B0, 'kappa')['d']
        pct, pcl = per_cohort(T, B0, 'kappa', 200), per_cohort(L[s], B0, 'kappa', 200)
        mx = max(abs(pct[c]['d'] - pcl[c]['d']) for c in COH)
        say(f'{s}: transductive {dt:+.4f}  loso {dl:+.4f}  max per-cohort difference {mx:.4f}')

    say('\n=== DATA REQUIREMENT (Figure 4), pooled ensemble kappa change by number of reference subjects')
    curve = []
    kc = {}
    for s in ('DA-G', 'DA-A', 'DA-MC'):
        parts = []
        for K in KS:
            T = E.table(da_files(s, PROT[K]), f'{s} {K}')
            r = boot(T, B0, 'kappa')
            kc[(s, K)] = r['d']
            if s != 'DA-A':
                curve.append(dict(series=s, K=K, dk=r['d'], n=r['n']))
            parts.append(f"{K}:{r['d']:+.4f} p={r['p']:.3f}" + (f" (unadapted {T.attrs['missing']})" if T.attrs['missing'] else ''))
        say(s, ' '.join(parts))
    T = E.table(da_files('DA-G', 'loso1'), 'DA-G 1')
    say('DA-G at one subject: acc pp', f"{boot(T, B0, 'accuracy')['d']:+.2f}")
    for K in ('1', '2', '5'):
        T = E.table(da_files('DA-MC', PROT[K]), f'DA-MC {K}')
        say(f'DA-MC loso{K}:', fmt(boot(T, B0, 'kappa')))
    T = E.table(da_files('DA-MC', 'loso1'), 'DA-MC 1')
    say('DA-MC one subject, per cohort:', pc_str(per_cohort(T, B0)))
    say('DA-MC one subject: within-10 pp', fmt(boot(T, B0, 'w10'), 2))
    for K in ('5', '15'):
        say(f"DA-MC at {K} subjects: {pct_of(kc[('DA-MC', K)], kc[('DA-MC', 'all')])} of its full gain;"
            f" DA-A {pct_of(kc[('DA-A', K)], kc[('DA-A', 'all')])}")

    say('\n=== REM-ONSET ENDPOINTS under adaptation (Table 6 within-10 column), ensemble')
    for s, T in L.items():
        say(f"{s}: within-10 {B0.w10.mean():.1f} -> {T.w10.mean():.1f}  d {fmt(boot(T, B0, 'w10'), 2)}  "
            f"false early {int(B0.early.sum())} -> {int(T.early.sum())}")

    say('\n=== DA-MC WITH EXPERT LABELS (Supplementary Table S5), ensemble')
    for K in KS:
        O = E.table(da_files('DA-MC expert', PROT[K]), f'DA-MC expert {K}')
        P = E.table(da_files('DA-MC', PROT[K]), f'DA-MC {K}')
        say(f"K={K:>3}: expert labels {fmt(boot(O, B0, 'kappa'))}  pseudo-labels {boot(P, B0, 'kappa')['d']:+.4f}  "
            f"paired {fmt(boot(O, P, 'kappa'))}")

    say('\n=== DA-O, classifier head fine-tuned on expert labels (halves protocol), ensemble')
    HB = E.table(lambda rid, f: f'{DA4}/fold{f}_{a.dao_tag}/baseline/proba/{rid}.npz', 'DA-O baseline')
    say('DA-O baseline ensemble kappa', round(HB.kappa.mean(), 4), '(unadapted ensemble', round(B0.kappa.mean(), 4), ')')
    res = {}
    for K in ('1', '2', '5', '10', '15', 'half'):
        T = E.table(dao_files(K), f'DA-O {K}')
        r = boot(T, HB, 'kappa')
        res[K] = r['d']
        curve.append(dict(series='DA-O', K='all' if K == 'half' else K, dk=r['d'], n=r['n']))
        say(f"K={K:>4}: {fmt(r)}  minus DA-MC (all subjects) {fmt(boot(T, L['DA-MC'], 'kappa'))}"
            f"  (unadapted in {T.attrs['missing']} recording-folds)")
    T = E.table(dao_files('half'), 'DA-O half')
    pc = per_cohort(T, HB)
    worst = min(pc.items(), key=lambda kv: kv[1]['d'])
    say(f"   half: acc pp {boot(T, HB, 'accuracy')['d']:+.2f}  F1 pp {boot(T, HB, 'f1_macro')['d']:+.2f}  "
        f"ROL AE {fmt(boot(T, HB, 'rol_ae'), 2)}  smallest cohort {worst[0]} {worst[1]['d']:+.4f}"
        + (' n.s.' if worst[1]['p'] >= 0.05 else ''))
    say('   half per cohort:', pc_str(pc))
    for K in ('5', '10', '15'):
        say(f"   K={K}: {pct_of(res[K], res["half"])} of the half-cohort value; {ratio(res[K], kc[('DA-MC', K)])} times DA-MC from the same {K} subjects")
    say('   per cohort by K (Supplementary Table S14):')
    pcK = pd.DataFrame({K: {c: v['d'] for c, v in per_cohort(E.table(dao_files(K), f'DA-O {K}'), HB, 'kappa', 200).items()}
                        for K in ('1', '2', '5', '10', '15', 'half')})
    pcK.loc['Pooled'] = [res[K] for K in pcK.columns]
    say(pcK.round(4).to_string())
    for K in ('1', '2', '5', '10', '15', 'half'):
        T = E.table(dao_files(K), f'DA-O {K}')
        say(f"   DA-O K={K:>4}: within-10 {HB.w10.mean():.1f} -> {T.w10.mean():.1f}  d {fmt(boot(T, HB, 'w10'), 2)}  "
            f"false early {int(HB.early.sum())} -> {int(T.early.sum())}")

    os.makedirs(f'{a.out}/analysis/paper/da_runs', exist_ok=True)
    pd.DataFrame(curve).to_csv(f'{a.out}/analysis/paper/da_runs/current_kcurve.csv', index=False)
    per = pd.concat([T.assign(condition=k) for k, T in E.cache.items()])
    per.to_csv(f'{a.out}/analysis/da_ensemble_perrec.csv', index=False)
    open(f'{a.out}/da_ensemble_paper.txt', 'w').write('\n'.join(out) + '\n')
    say('\nwrote', f'{a.out}/da_ensemble_paper.txt')


if __name__ == '__main__':
    main()
