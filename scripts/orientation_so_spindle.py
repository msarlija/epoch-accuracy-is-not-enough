# -*- coding: utf-8 -*-
"""Slow-oscillation / spindle coupling as an orientation test.

Sleep spindles (11-16 Hz) are grouped on the UP state of the slow oscillation,
which in the standard sign convention (electrode minus mastoid, positive up) is
the POSITIVE half-wave. So, per recording-channel in expert N3, the 150 largest
positive peaks and the 150 largest negative troughs of the 0.3-2 Hz band are
taken separately, and both the raw signal and the z-scored spindle envelope
(Hilbert amplitude of 11-16 Hz, z-scored over N3) are averaged +/-1.5 s around
them. Standard orientation: spindle power is higher around the positive peaks;
inverted: around the negative troughs. Unlike skew, this does not assume which
half-wave is larger.

Index per dataset: mean spindle z within +/-0.3 s of the positive peaks minus
the same around the negative troughs (> 0 standard, < 0 inverted).

    python scripts/orientation_so_spindle.py --data-dir <corpus> --out-dir <dir> --apply-sign --split-hpap \
        --per-cohort 30 --datasets CFS YSYW HPAP BOAS MASS MrOS SOF WSC DODO DODH MASS-SS1 ANPHY DREAMS-P DREAMS-S
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import argrelextrema, butter, hilbert, sosfiltfilt  # noqa: E402

FS, EP, HALF, TOP = 100.0, 3000, 150, 150
SO = butter(4, [0.3, 2.0], 'bandpass', fs=FS, output='sos')
SIG = butter(4, [11.0, 16.0], 'bandpass', fs=FS, output='sos')
W = int(0.3 * FS)


def per_channel(raw, n3_idx):
    so = sosfiltfilt(SO, raw)
    env = np.abs(hilbert(sosfiltfilt(SIG, raw)))
    keep = np.zeros(len(raw), bool)
    keep[n3_idx] = True
    good = keep & (np.abs(raw) < 300)
    mu, sd = env[good].mean(), env[good].std() + 1e-9
    envz = (env - mu) / sd
    out = {}
    for name, ext in (('peak', argrelextrema(so, np.greater)[0]), ('trough', argrelextrema(so, np.less)[0])):
        ext = ext[keep[ext] & (ext > HALF) & (ext < len(raw) - HALF) & (np.abs(so[ext]) < 300)]
        if len(ext) < 20:
            return None
        ext = ext[np.argsort(-np.abs(so[ext]))][:TOP]
        out[name + '_wave'] = np.stack([raw[e - HALF:e + HALF] - np.median(raw[e - HALF:e + HALF]) for e in ext]).mean(0)
        out[name + '_sigma'] = np.stack([envz[e - HALF:e + HALF] for e in ext]).mean(0)
        out[name + '_amp'] = float(np.mean(np.abs(so[ext])))
    out['index'] = float(out['peak_sigma'][HALF - W:HALF + W].mean() - out['trough_sigma'][HALF - W:HALF + W].mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--datasets', nargs='+',
                    default=['DODO', 'DODH', 'MASS-SS1', 'WSC', 'YSYW', 'MASS', 'ANPHY', 'CFS'])
    ap.add_argument('--per-cohort', type=int, default=15)
    ap.add_argument('--apply-sign', action='store_true',
                    help="multiply by the manifest's sign column, i.e. test the signal as the model reads it")
    ap.add_argument('--split-hpap', action='store_true',
                    help='report HomePAP nights with a recorded FPz and nights with stored M1/M2 separately')
    a = ap.parse_args()
    m = pd.read_csv(f'{a.data_dir}/manifest.csv')
    m = m[(m.qc_pass == 1) & m.cohort.isin(a.datasets)].copy()
    if a.split_hpap and 'HPAP' in a.datasets:
        fpz = m.cohort.eq('HPAP') & m.channel_names.astype(str).str.contains('FPz')
        m.loc[fpz, 'cohort'] = 'HPAP-FPz'
        m.loc[m.cohort.eq('HPAP'), 'cohort'] = 'HPAP-M'
        i = a.datasets.index('HPAP')
        a.datasets[i:i + 1] = ['HPAP-FPz', 'HPAP-M']
    res = {}
    for coh, g in m.groupby('cohort'):
        for _, r in g.sample(min(a.per_cohort, len(g)), random_state=0).iterrows():
            sign = float(r['sign']) if a.apply_sign and 'sign' in r and pd.notna(r['sign']) else 1.0
            x = np.load(f'{a.data_dir}/{r.rec_id}_x.npy', mmap_mode='r')
            y = np.load(f'{a.data_dir}/{r.rec_id}_y.npy').astype(int)
            n3 = np.flatnonzero(y == 3)
            n3 = n3[(n3 + 1) * EP <= x.shape[0]][:300]
            if len(n3) < 10:
                continue
            idx = (n3[:, None] * EP + np.arange(EP)).ravel()
            for c in [i for i, v in enumerate(json.loads(r.qc_channel_flags)) if v == 1] or [0]:
                o = per_channel(sign * np.asarray(x[:, c], np.float64), idx)
                if o:
                    res.setdefault(coh, []).append(o)
    order = [d for d in a.datasets if d in res]
    S = pd.DataFrame([dict(dataset=d, channels=len(res[d]),
                           spindle_index=np.mean([o['index'] for o in res[d]]),
                           channels_index_positive=np.mean([o['index'] > 0 for o in res[d]]),
                           peak_amp_uV=np.mean([o['peak_amp'] for o in res[d]]),
                           trough_amp_uV=np.mean([o['trough_amp'] for o in res[d]])) for d in order]).set_index('dataset')
    S['orientation (spindles)'] = np.where(S.spindle_index > 0, 'standard', 'inverted')
    S.round(3).to_csv(f'{a.out_dir}/so_spindle_coupling.csv')
    pd.set_option('display.width', 200)
    print(S.round(3).to_string())

    t = (np.arange(2 * HALF) - HALF) / FS
    fig, axes = plt.subplots(len(order), 2, figsize=(10, 2.0 * len(order)), sharex=True)
    for i, d in enumerate(order):
        R = res[d]
        for k, col, lab in (('trough', '#0072B2', 'locked to negative troughs'),
                            ('peak', '#D55E00', 'locked to positive peaks')):
            axes[i, 0].plot(t, np.mean([o[k + '_wave'] for o in R], 0), color=col, lw=1.6, label=lab)
            axes[i, 1].plot(t, np.mean([o[k + '_sigma'] for o in R], 0), color=col, lw=1.6, label=lab)
        for ax in axes[i]:
            ax.axvline(0, color='0.6', lw=0.5, ls=':')
            ax.axhline(0, color='0.85', lw=0.5)
            ax.tick_params(labelsize=7)
        axes[i, 0].set_ylabel(f'{d}\nuV (positive up)', fontsize=8)
        axes[i, 1].set_ylabel('spindle\nenvelope (z)', fontsize=8)
        axes[i, 1].set_title(f'spindle index {S.loc[d, "spindle_index"]:+.2f} -> {S.loc[d, "orientation (spindles)"]}',
                             fontsize=8, loc='left')
    axes[0, 0].set_title('average slow wave', fontsize=9, loc='left')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower right')
    axes[-1, 0].set_xlabel('s from the slow-wave extremum', fontsize=8)
    axes[-1, 1].set_xlabel('s from the slow-wave extremum', fontsize=8)
    fig.suptitle('N3 slow waves and spindle activity, as stored in the corpus. Spindles ride on the UP state: in the '
                 'standard orientation\nspindle power is higher around the POSITIVE peaks (vermillion) than the '
                 'negative troughs (blue).', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f'{a.out_dir}/so_spindle_coupling.png', dpi=130)
    print('wrote', f'{a.out_dir}/so_spindle_coupling.png')


if __name__ == '__main__':
    main()
