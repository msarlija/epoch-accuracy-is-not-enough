# -*- coding: utf-8 -*-
"""Re-extract the ANPHY-Sleep prefrontal EEG with a mastoid-side reference.

The released ANPHY EEG is referential to the amplifier's system reference,
which sits centrally (stored C3 and C4 mirror each other, r = -0.97 to -0.999,
i.e. the reference is close to their mean). Fp1 and Fp2 as stored are
therefore frontal-minus-central derivations, unlike every other cohort of the
corpus. This script rebuilds them against the contralateral mastoid-side
electrodes, as in the AASM recommended montage:

    Fp1-TP10 = Fp1 - TP10        Fp2-TP9 = Fp2 - TP9

(TP9 and TP10 are the 10-10 positions over the left and right mastoids; the
common system reference cancels in the difference.)

For every subject the archive is streamed from OSF (doi 10.17605/OSF.IO/R26FH)
and decompressed on the fly, so neither the 2.5-3.5 GB zip nor the ~5 GB EDF is
ever written to disk. Only Fp1, Fp2, TP9 and TP10 are kept from each 1 s data
record. The two derivations are resampled from 1000 to 250 Hz with a polyphase
FIR filter (scipy.signal.resample_poly, which low-pass filters before
decimating, so there is no aliasing), and written as a small EDF:

    <ANPHY>/<subject>/<subject>_Fp-TP_250Hz.edf

together with the scoring files contained in the archive. A log of every
subject is appended to <ANPHY>/reextract_log.csv. Subjects whose output exists
are skipped, so the script can be re-run after an interruption.

    python reextract_anphy.py                 # all subjects
    python reextract_anphy.py EPCTL04 EPCTL05 # selected subjects

Requires numpy and scipy only. Writes to $SLEEP_RAW_ROOT/ANPHY.

Notes from the October 2026 run: 28 of the 29 subjects of the subjects table
are on OSF (EPCTL08 is no longer in the archive, so the cohort has 28
recordings). EPCTL21 and EPCTL25 label their channels 'Fp1-Ref' etc., which
the channel lookup accepts.
"""

import csv
import datetime
import json
import os
import re
import struct
import sys
import time
import urllib.request
import zlib
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

# the ANPHY folder of the raw-data root (as in configs/preprocess.yaml); the
# script directory when SLEEP_RAW_ROOT is not set
ROOT = (os.path.join(os.environ['SLEEP_RAW_ROOT'], 'ANPHY') if os.environ.get('SLEEP_RAW_ROOT')
        else os.path.dirname(os.path.abspath(__file__)))
OSF_API = 'https://api.osf.io/v2/nodes/r26fh/files/osfstorage/'
SUBJECTS_CSV = os.path.join(ROOT, 'Details information for healthy subjects.csv')
LOG = os.path.join(ROOT, 'reextract_log.csv')
TARGET_FS = 250
PAIRS = (('Fp1', 'TP10'), ('Fp2', 'TP9'))          # contralateral mastoid-side reference
KEEP = ('Fp1', 'Fp2', 'TP9', 'TP10')
CHUNK = 8 * 1024 * 1024


# ----------------------------------------------------------------- OSF listing
def osf_archives():
    """{subject id: download url} for the subject archives on OSF."""
    out, url = {}, OSF_API
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for x in d['data']:
            name = x['attributes']['name']
            m = re.match(r'(EPCTL\d+).*\.zip$', name)
            if m and x['attributes']['kind'] == 'file':
                out[m.group(1)] = (name, x['links']['download'], x['attributes']['size'])
        url = d['links'].get('next')
    return out


# ----------------------------------------------------------------- EDF reader (streaming)
class EDFStream:
    """Consumes the bytes of one EDF as they are decompressed and keeps only
    the channels in KEEP (in uV) and each record's onset from the annotations."""

    def __init__(self, name):
        self.name, self.buf, self.hdr = name, bytearray(), None
        self.data = {c: [] for c in KEEP}
        self.onsets = []

    def _parse_header(self):
        b = bytes(self.buf)
        ns = int(b[252:256])
        need = 256 + 256 * ns
        if len(b) < need:
            return False

        def field(off, width):
            base = 256 + off * ns
            return [b[base + i * width: base + (i + 1) * width].decode('latin-1').strip() for i in range(ns)]

        lab = field(0, 16)
        dim = field(96, 8)
        pmin = np.array(field(104, 8), float); pmax = np.array(field(112, 8), float)
        dmin = np.array(field(120, 8), float); dmax = np.array(field(128, 8), float)
        spr = np.array([int(v) for v in field(216, 8)])
        upper = [s.upper() for s in lab]
        # some recordings (e.g. EPCTL21) label the referential channels 'Fp1-Ref'
        bare = [u[:-4] if u.endswith('-REF') else u for u in upper]
        idx = {}
        for c in KEEP:
            if c.upper() not in bare:
                raise RuntimeError(f'{self.name}: channel {c} not found; labels {lab}')
            idx[c] = bare.index(c.upper())
        ia = upper.index('EDF ANNOTATIONS') if 'EDF ANNOTATIONS' in upper else None
        scale = {'UV': 1.0, 'µV': 1.0, 'MV': 1e3, 'V': 1e6}
        conv = {}
        for c, i in idx.items():
            unit = dim[i].upper().replace('µ', 'U')
            if unit not in scale:
                raise RuntimeError(f'{self.name}: unexpected unit {dim[i]!r} for {c}')
            g = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
            conv[c] = (g * scale[unit], (pmin[i] - dmin[i] * g) * scale[unit])
        dur = float(b[244:252])
        fs = {c: spr[i] / dur for c, i in idx.items()}
        assert len(set(fs.values())) == 1, fs
        offs = np.concatenate([[0], np.cumsum(spr)]) * 2
        self.hdr = dict(ns=ns, lab=lab, idx=idx, ia=ia, conv=conv, dur=dur, fs=list(fs.values())[0],
                        offs=offs, recbytes=int(spr.sum()) * 2, reserved=b[192:236].decode('latin-1').strip(),
                        start_date=b[168:176].decode('latin-1'), start_time=b[176:184].decode('latin-1'),
                        n_records=int(b[236:244]))
        del self.buf[:need]
        return True

    def feed(self, chunk):
        self.buf += chunk
        if self.hdr is None and (len(self.buf) < 256 or not self._parse_header()):
            return
        h = self.hdr
        rb = h['recbytes']
        n = len(self.buf) // rb
        if n == 0:
            return
        block = np.frombuffer(bytes(self.buf[:n * rb]), dtype='<i2').reshape(n, rb // 2)
        for c, i in h['idx'].items():
            a, z = h['offs'][i] // 2, h['offs'][i + 1] // 2
            g, o = h['conv'][c]
            self.data[c].append((block[:, a:z].astype(np.float32) * g + o).reshape(-1))
        if h['ia'] is not None:
            a, z = h['offs'][h['ia']], h['offs'][h['ia'] + 1]
            for r in range(n):
                tal = bytes(self.buf[r * rb + a: r * rb + z]).split(b'\x14', 1)[0].decode('latin-1')
                try:
                    self.onsets.append(float(tal))
                except ValueError:
                    self.onsets.append(np.nan)
        del self.buf[:n * rb]

    def result(self):
        h = self.hdr
        x = {c: np.concatenate(v) for c, v in self.data.items()}
        on = np.asarray(self.onsets)
        gaps = int(np.sum(np.abs(np.diff(on) - h['dur']) > 1e-3)) if len(on) > 1 else 0
        return x, h, gaps, len(on)


# ----------------------------------------------------------------- zip reader (streaming)
def stream_zip(url, on_file_start, on_bytes, on_file_end):
    """Parse a zip archive from an HTTP stream, decompressing each entry."""
    req = urllib.request.Request(url, headers={'User-Agent': 'anphy-reextract'})
    with urllib.request.urlopen(req, timeout=120) as r:
        pending = b''

        def read(n):
            nonlocal pending
            while len(pending) < n:
                more = r.read(CHUNK)
                if not more:
                    break
                pending += more
            out, pending = pending[:n], pending[n:]
            return out

        while True:
            sig = read(4)
            if sig != b'PK\x03\x04':
                break                                  # central directory: done
            hdr = read(26)
            ver, flags, method, _t, _d, _crc, csize, usize, fnl, exl = struct.unpack('<HHHHHIIIHH', hdr)
            name = read(fnl).decode('utf-8', 'replace')
            read(exl)
            on_file_start(name)
            if method == 0 and not flags & 0x08:
                left = csize
                while left:
                    piece = read(min(left, CHUNK)); left -= len(piece)
                    on_bytes(name, piece)
            elif method == 8:
                z = zlib.decompressobj(-15)
                while not z.eof:
                    piece = pending or r.read(CHUNK)
                    pending = b''
                    if not piece:
                        raise RuntimeError(f'stream ended inside {name}')
                    out = z.decompress(piece)
                    if out:
                        on_bytes(name, out)
                pending = z.unused_data + pending
                if flags & 0x08:                       # data descriptor follows
                    d = read(4)
                    read(20 if d == b'PK\x07\x08' and usize == 0xFFFFFFFF else (12 if d == b'PK\x07\x08' else 8))
            else:
                raise RuntimeError(f'unsupported compression {method} for {name}')
            on_file_end(name)


# ----------------------------------------------------------------- EDF writer
def write_edf(path, signals, labels, fs, start_date, start_time, note):
    """Plain EDF, 1 s records, int16, physical range fitted to each signal."""
    n = len(signals[0])
    spr = int(fs)
    nrec = int(np.ceil(n / spr))
    pad = nrec * spr - n
    ns = len(signals)
    phys = []
    for x in signals:
        m = float(np.nanmax(np.abs(x)))
        phys.append(max(500.0, np.ceil(m / 100.0) * 100.0))

    def f(s, w):
        s = str(s)[:w]
        return s + ' ' * (w - len(s))

    h = f('0', 8) + f('X X X X', 80) + f(f'Startdate {start_date} X X {note}', 80)
    h += f(start_date, 8) + f(start_time, 8) + f(256 + 256 * ns, 8) + f('', 44)
    h += f(nrec, 8) + f(1, 8) + f(ns, 4)
    h += ''.join(f(l, 16) for l in labels)
    h += ''.join(f('AgAgCl electrode', 80) for _ in labels)
    h += ''.join(f('uV', 8) for _ in labels)
    h += ''.join(f(f'{-p:.0f}', 8) for p in phys)
    h += ''.join(f(f'{p:.0f}', 8) for p in phys)
    h += ''.join(f(-32768, 8) for _ in labels)
    h += ''.join(f(32767, 8) for _ in labels)
    h += ''.join(f(f'rereferenced; polyphase resampled 1000->{fs:.0f} Hz', 80) for _ in labels)
    h += ''.join(f(spr, 8) for _ in labels)
    h += ''.join(f('', 32) for _ in labels)
    assert len(h) == 256 + 256 * ns
    dig = []
    for x, p in zip(signals, phys):
        y = np.concatenate([x, np.zeros(pad, np.float32)])
        d = np.clip(np.round((y + p) / (2 * p) * 65535 - 32768), -32768, 32767).astype('<i2')
        dig.append(d.reshape(nrec, spr))
    body = np.stack(dig, axis=1).reshape(-1)
    with open(path + '.part', 'wb') as fh:
        fh.write(h.encode('latin-1'))
        fh.write(body.tobytes())
    os.replace(path + '.part', path)
    return phys, pad


# ----------------------------------------------------------------- one subject
def process(subject, name, url):
    out_dir = os.path.join(ROOT, subject)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{subject}_Fp-TP_250Hz.edf')
    edfs, small, current = {}, {}, {}

    def start(n):
        if n.lower().endswith('.edf'):
            current[n] = EDFStream(n)
            edfs[n] = current[n]
        elif not n.endswith('/'):
            small[n] = bytearray()

    def data(n, piece):
        if n in edfs:
            edfs[n].feed(piece)
        elif n in small:
            small[n] += piece

    t0 = time.time()
    stream_zip(url, start, data, lambda n: None)
    if not edfs:
        raise RuntimeError('no EDF in archive')
    # a corrected file ("... - fixed.edf") takes precedence over the original
    pick = sorted(edfs, key=lambda n: ('fixed' not in n.lower(), n))[0]
    x, h, gaps, nrec = edfs[pick].result()
    if gaps:
        raise RuntimeError(f'{pick}: {gaps} discontinuities between data records ({h["reserved"]})')
    ratio = Fraction(TARGET_FS, int(round(h['fs']))).limit_denominator(1000)
    sig, labels = [], []
    for a, ref in PAIRS:
        y = x[a].astype(np.float64) - x[ref].astype(np.float64)
        sig.append(resample_poly(y, ratio.numerator, ratio.denominator).astype(np.float32))
        labels.append(f'{a}-{ref}')
    phys, pad = write_edf(out, sig, labels, TARGET_FS, h['start_date'], h['start_time'],
                          f'ANPHY {subject} Fp-TP rereferenced')
    saved = []
    for n, content in small.items():
        if n.lower().endswith(('.txt', '.csv', '.tsv', '.xml', '.json')):
            dst = os.path.join(out_dir, os.path.basename(n))
            with open(dst, 'wb') as fh:
                fh.write(bytes(content))
            saved.append(os.path.basename(n))
    q = [np.subtract(*np.percentile(s, [75, 25])) for s in sig]
    return dict(subject=subject, archive=name, edf=os.path.basename(pick), edf_type=h['reserved'],
                fs_in=h['fs'], records=nrec, hours=round(nrec * h['dur'] / 3600, 2), gaps=gaps,
                fs_out=TARGET_FS, pad_samples=pad, phys_range_uV=';'.join(f'{p:.0f}' for p in phys),
                iqr_uV=';'.join(f'{v:.1f}' for v in q), corr=round(float(np.corrcoef(sig[0], sig[1])[0, 1]), 3),
                other_files=';'.join(saved), seconds=round(time.time() - t0),
                done=datetime.datetime.now().isoformat(timespec='seconds'))


def log(row):
    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    with open(SUBJECTS_CSV, newline='', encoding='utf-8-sig') as fh:
        subjects = [r['Subjects ID'].strip() for r in csv.DictReader(fh) if r.get('Subjects ID')]
    if len(sys.argv) > 1:
        subjects = [s for s in subjects if s in sys.argv[1:]]
    archives = osf_archives()
    for s in subjects:
        out = os.path.join(ROOT, s, f'{s}_Fp-TP_250Hz.edf')
        if os.path.exists(out):
            print(f'{s}: done already, skipped', flush=True)
            continue
        if s not in archives:
            print(f'{s}: no archive on OSF, skipped', flush=True)
            log(dict(subject=s, archive='', edf='', edf_type='', fs_in='', records='', hours='', gaps='',
                     fs_out='', pad_samples='', phys_range_uV='', iqr_uV='', corr='', other_files='',
                     seconds='', done='missing on OSF'))
            continue
        name, url, size = archives[s]
        for attempt in (1, 2, 3):
            try:
                print(f'{s}: streaming {name} ({size / 1e9:.1f} GB), attempt {attempt}', flush=True)
                row = process(s, name, url)
                log(row)
                print(f'{s}: {row["hours"]} h, IQR {row["iqr_uV"]} uV, r(Fp1-TP10, Fp2-TP9) {row["corr"]}, '
                      f'files {row["other_files"]}, {row["seconds"]} s', flush=True)
                break
            except Exception as e:                    # network errors: retry the subject
                print(f'{s}: attempt {attempt} failed: {e}', flush=True)
                time.sleep(30)
        else:
            print(f'{s}: FAILED after 3 attempts', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
