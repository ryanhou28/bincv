#!/usr/bin/env python3
"""Aggregate several runs of the CUDA benchmarks and apply the project's rule.

WHY THIS EXISTS.  benchmark/measure_util.hpp states the rule this project
decides performance questions by:

    A difference smaller than the spread is a null result, and a null result
    is a result.  The spread bounds WITHIN-run noise only; run-to-run scatter
    is a separate and sometimes larger number, and an entry that calls a
    difference real should clear the larger of the two.

A benchmark process can see only the first half.  The second half is the
scatter of a benchmark's MEDIANS across independent processes, which means
several runs and something that totals them.  Until this script there was
nothing: every cross-run figure in docs/reports/cuda.md was produced by a
throwaway, so the numbers were reproducible in principle and not in practice,
and the stronger half of the rule was never actually applied to any of them.

WHAT IT READS.  Two machine-readable lines the benchmarks print beside their
prose, in any number of files:

  PAIRED|scope|armA|armB|...   emitted by cudabench::printPaired, so every
                              paired comparison in the backend has one.
  ROW,key,geom,...            emitted by the families that key their rows by
                              hand (the role comparison, the packer).

Both carry both arms' min/median/max, the per-round ratio's min/median/max,
the geometric mean, the sign split and the separation bit.

WHAT IT COMPUTES, per key, over the runs that carry it:

  median      the median of the per-run median ratios -- the value to quote.
  within      the median of the per-run per-round SWINGS (ratio max / min):
              the within-run half of the rule.
  r2r         max / min of the per-run medians: the run-to-run half, which is
              the number one process cannot see.
  bar         the larger of those two, which is what the rule says to clear.
  diff        max(median, 1 / median) -- how many times apart the arms are.
  verdict     RESULT when diff > bar, otherwise NULL RESULT, which is a result.

EVERYTHING IS A FACTOR, never a percentage, and that is not presentation.
|median - 1| as a percentage cannot exceed 100% for the arm that is FASTER
however far ahead it is, while (max - min) / median has no ceiling -- so the
same rounds come out a result or a null depending only on which arm went in
the denominator.  backends/cuda/benchmark/paired_stats.hpp carries the worked
example and the test that pins it.

SEPARATION IS REPORTED AND DOES NOT DECIDE.  The `sep` column counts the runs
whose two arms had disjoint sample ranges.  It is a strong fact when it is
high and it is not the verdict: one round slow in BOTH arms overlaps the
ranges while leaving every per-round ratio untouched.

USAGE
    scripts/aggregate_cuda_runs.py <dir-or-file> [<dir-or-file> ...] [-k REGEX]

Each input may be a directory of run outputs or a single file.  A run's family
is taken from the file name up to the last underscore, so `lk_3.txt` and
`lk_4.txt` aggregate together and `stereo_3.txt` stays separate.
"""

import argparse
import os
import re
import sys
from statistics import median


def _families(paths):
    """Yields (family, path) for every run file under the given paths."""
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if os.path.isfile(full) and not name.endswith(('.py', '.sh', '.log')):
                    yield name.rsplit('_', 1)[0], full
        else:
            yield os.path.basename(p).rsplit('_', 1)[0], p


def _sample(ratio_min, ratio_med, ratio_max, wins_a, wins_b, separated):
    """One process's contribution to one key."""
    if ratio_med <= 0.0 or ratio_min <= 0.0:
        return None
    return dict(med=ratio_med,
                swing=ratio_max / ratio_min,
                wa=wins_a, wb=wins_b, sep=separated)


def collect(paths):
    """key -> [per-run sample].  A key is (family, name, scope)."""
    runs = {}
    for fam, path in _families(paths):
        with open(path, errors='replace') as fh:
            for line in fh:
                line = line.rstrip('\n')
                s = None
                if line.startswith('PAIRED|'):
                    f = line.split('|')
                    if len(f) >= 22:
                        try:
                            s = (fam, f[2].strip() + '  vs  ' + f[3].strip(), f[1].strip())
                            v = _sample(float(f[10]), float(f[11]), float(f[12]),
                                        int(f[15]), int(f[16]), int(f[21]))
                        except ValueError:
                            s = None
                elif line.startswith('ROW,'):
                    f = line.split(',')
                    if len(f) >= 21:
                        try:
                            s = (fam, f[1], f[2])
                            v = _sample(float(f[9]), float(f[10]), float(f[11]),
                                        int(f[15]), int(f[16]), int(f[12]))
                        except ValueError:
                            s = None
                if s is not None and v is not None:
                    runs.setdefault(s, []).append(v)
    return runs


def judge(runs, min_runs=2):
    """Applies measure_util.hpp's rule to each key, both halves."""
    out = []
    for key, rs in sorted(runs.items()):
        if len(rs) < min_runs:
            continue
        meds = [r['med'] for r in rs]
        quoted = median(meds)
        r2r = max(meds) / min(meds)
        within = median([r['swing'] for r in rs])
        diff = max(quoted, 1.0 / quoted)
        bar = max(within, r2r)
        out.append(dict(key=key, n=len(rs), median=quoted,
                        lo=min(meds), hi=max(meds), within=within, r2r=r2r,
                        diff=diff, bar=bar, result=diff > bar,
                        sep=sum(r['sep'] for r in rs),
                        wa=sum(r['wa'] for r in rs), wb=sum(r['wb'] for r in rs)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='run outputs: directories or files')
    ap.add_argument('-k', '--key', default='', help='regex, matched against the key')
    ap.add_argument('-n', '--min-runs', type=int, default=2,
                    help='ignore keys seen in fewer runs than this (default 2)')
    args = ap.parse_args(argv)

    rows = judge(collect(args.paths), args.min_runs)
    if args.key:
        pat = re.compile(args.key, re.I)
        rows = [r for r in rows if pat.search(' / '.join(r['key']))]
    if not rows:
        print('no rows matched', file=sys.stderr)
        return 1

    width = min(86, max(len(' / '.join(r['key'])) for r in rows))
    print(f"{'key':<{width}} {'n':>2} {'median':>8} {'range':>19} {'diff':>7} "
          f"{'within':>7} {'r2r':>6} {'bar':>7} {'verdict':<8} {'sep':>6} {'sign':>9}")
    for r in rows:
        k = ' / '.join(r['key'])
        print(f"{k[:width]:<{width}} {r['n']:>2} {r['median']:>8.4f} "
              f"{r['lo']:>8.4f}-{r['hi']:<10.4f} {r['diff']:>7.3f} "
              f"{r['within']:>7.3f} {r['r2r']:>6.3f} {r['bar']:>7.3f} "
              f"{'RESULT' if r['result'] else 'NULL':<8} "
              f"{r['sep']:>2}/{r['n']:<3} {r['wa']:>4}-{r['wb']:<4}")
    print("\n  median  the value to quote: the median of the per-run median ratios.")
    print("  diff    how many times apart the arms are, max(median, 1/median).")
    print("  within  median per-run swing of the per-round ratio (max/min).")
    print("  r2r     run-to-run scatter: max/min of the per-run medians.")
    print("  bar     the larger of within and r2r -- measure_util.hpp's rule.")
    print("  sep     runs whose arms had DISJOINT ranges. A fact, not the verdict.")
    print("  sign    paired rounds favouring arm A - arm B, summed over runs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
