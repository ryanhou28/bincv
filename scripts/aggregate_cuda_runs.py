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
  magnitude   min..max per-round factor, over EVERY round of EVERY run, in the
              established direction.  Printed only when one is established.
  p           exact two-sided sign-test p over all of those rounds.

THE VERDICT HAS THREE VALUES, not two (owner's ruling, 2026-09-19).

  DIRECTION ESTABLISHED   no round, in any run, crossed 1.00x.  The sign of the
                          difference is settled by the observations; the spread
                          then bounds only HOW MUCH the winning arm wins by, so
                          the row is quoted as a RANGE with its sign count --
                          "faster in 105 of 105 rounds, by 1.62x to 6.71x,
                          median 2.18x" -- rather than collapsed to one number.
  RESULT                  diff > bar.  The old predicate, unchanged.
  NULL RESULT             neither, and still a result.

A row can be both of the first two and the table says so, because they are
different statements: "which arm is ahead is not in doubt" and "the distance
between them exceeds the noise".  The `old` column is what the two-valued rule
said about the SAME rounds, so a re-judging pass can be read off one table.

NOTHING GATES ON A ROUND COUNT.  Two unanimous rounds and a hundred unanimous
rounds both satisfy "no round crossed", and a minimum-round-count rule would be
a project-wide "X is enough" bar invented here -- which CLAUDE.md forbids.  The
exact sign-test p is printed instead, so the reader can tell them apart from a
number that is on the page.

A TIE BREAKS THE DIRECTION VERDICT.  A round whose arms time identically
favours neither, and the sentence this verdict licenses is "faster in N of N
rounds".  Excluding ties would make the verdict easier to earn the coarser the
clock is, and the tie bucket also holds rounds that were not measurements at
all.  The sign TEST still excludes them from n, which is what a sign test does.

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

ONE FILE MAY HOLD SEVERAL RUNS, and a committed sweep does.
`scripts/run_cuda_launches.sh` writes a whole sweep as a single stamped log --
one artifact per sweep, with a `### run N` line between launches -- because the
staleness gate reads one header per log and a directory of unstamped files
names no commit.  So a file is split on those markers before anything is
parsed, using the same splitter `scripts/aggregate_launches.py` uses on the
host logs.  Without that split a seven-process sweep would read as ONE run: the
run-to-run half of the rule would collapse to 1.000 and every row would look
far better established than the measurements support.  A file with no markers
is one run, which is what a directory of per-process outputs still is.
"""

import argparse
import os
import re
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# One definition of what a launch marker looks like, shared with the host
# aggregator rather than copied: the two scripts read the same `### run N`
# convention out of logs written by two runners, and a second spelling of it
# would silently read a seven-process sweep as one process.
from aggregate_launches import split_launches  # noqa: E402


def _family_of(path):
    """The group a file's runs belong to.

    A whole sweep in one file is its own family, named for the sweep. Otherwise
    the file is one process of a family and the name carries the index, so
    `role_3.txt` and `role_4.txt` join and `stereo_3.txt` stays apart.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ('-cuda-launches', '-launches'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem.rsplit('_', 1)[0]


def _families(paths):
    """Yields (family, path) for every run file under the given paths."""
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if not os.path.isfile(full) or name.endswith(('.py', '.sh')):
                    continue
                # A `.log` in a run directory is a build or gate transcript, not
                # a run -- except a committed sweep, which is a whole family in
                # one file and is named as one.
                if name.endswith('.log') and not name.endswith('-launches.log'):
                    continue
                yield _family_of(full), full
        else:
            yield _family_of(p), p


def sign_test_two_sided_p(wins_a, wins_b):
    """Exact two-sided sign-test p, the same arithmetic paired_stats.hpp does.

    2 * sum(C(n, i), i = 0..min(wins_a, wins_b)) / 2^n, clamped at 1.  Ties are
    excluded from n, which is what a sign test does with them -- and is a
    statement about a probability model with no third outcome, not about which
    observations the DIRECTION verdict may look at.  Reported, never gated on.
    """
    if wins_a < 0 or wins_b < 0:
        return 1.0
    n = wins_a + wins_b
    if n <= 0:
        return 1.0
    k = min(wins_a, wins_b)
    term = 0.5 ** n
    tail = term
    for i in range(1, k + 1):
        term *= (n - i + 1) / i
        tail += term
    return min(1.0, 2.0 * tail)


def _sample(ratio_min, ratio_med, ratio_max, wins_a, wins_b, separated,
            rounds=0, tied=0):
    """One process's contribution to one key.

    `rounds` and `tied` are carried because the direction verdict is a claim
    about EVERY round of EVERY run -- 105 rounds from seven processes -- and no
    single process can see that.  They default to 0 so a run file written
    before those columns existed still aggregates on everything else; such a
    row simply cannot establish a direction, which is the safe direction for a
    missing column to fail in.
    """
    if ratio_med <= 0.0 or ratio_min <= 0.0:
        return None
    return dict(med=ratio_med, lo=ratio_min, hi=ratio_max,
                swing=ratio_max / ratio_min,
                wa=wins_a, wb=wins_b, sep=separated,
                rounds=rounds, tied=tied)


def _parse_run(fam, lines, runs):
    """One process's output -> its samples, appended to `runs`."""
    for line in lines:
        line = line.rstrip('\n')
        s = None
        if line.startswith('PAIRED|'):
            f = line.split('|')
            if len(f) >= 22:
                try:
                    s = (fam, f[2].strip() + '  vs  ' + f[3].strip(), f[1].strip())
                    v = _sample(float(f[10]), float(f[11]), float(f[12]),
                                int(f[15]), int(f[16]), int(f[21]),
                                int(f[14]), int(f[17]))
                except ValueError:
                    s = None
        elif line.startswith('ROW,'):
            f = line.split(',')
            if len(f) >= 21:
                try:
                    s = (fam, f[1], f[2])
                    v = _sample(float(f[9]), float(f[10]), float(f[11]),
                                int(f[15]), int(f[16]), int(f[12]),
                                int(f[13]), int(f[17]))
                except ValueError:
                    s = None
        if s is not None and v is not None:
            runs.setdefault(s, []).append(v)
    return runs


def collect(paths):
    """key -> [per-run sample].  A key is (family, name, scope).

    A file is split into launches first, so a committed sweep contributes the
    seven samples it holds rather than one -- see the header.
    """
    runs = {}
    for fam, path in _families(paths):
        with open(path, errors='replace') as fh:
            text = fh.read()
        for _label, lines in split_launches(text):
            _parse_run(fam, lines, runs)
    return runs


def judge(runs, min_runs=2):
    """Applies the project's rule to each key: both halves, all three values."""
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

        # THE DIRECTION HALF, pooled over every round of every run, because
        # that is the population the claim is about.  A key seen in seven runs
        # of fifteen rounds is 105 paired observations, and unanimity over them
        # is a fact no single process could report.
        wa = sum(r['wa'] for r in rs)
        wb = sum(r['wb'] for r in rs)
        tied = sum(r['tied'] for r in rs)
        rounds = sum(r['rounds'] for r in rs)
        # rounds == 0 means the columns were absent, not that nothing ran; such
        # a row cannot establish a direction, which is where a missing column
        # should fail.
        established = rounds > 0 and tied == 0 and (wa + wb) > 0 and (wa == 0 or wb == 0)
        if established and wa == 0:
            # Every round had B faster (ratio below 1): report the magnitude the
            # way up a reader wants it, as how much B won by.
            mag_lo, mag_hi = 1.0 / max(r['hi'] for r in rs), 1.0 / min(r['lo'] for r in rs)
        elif established:
            mag_lo, mag_hi = min(r['lo'] for r in rs), max(r['hi'] for r in rs)
        else:
            mag_lo = mag_hi = 0.0

        out.append(dict(key=key, n=len(rs), median=quoted,
                        lo=min(meds), hi=max(meds), within=within, r2r=r2r,
                        diff=diff, bar=bar, result=diff > bar,
                        established=established, mag_lo=mag_lo, mag_hi=mag_hi,
                        rounds=rounds, tied=tied, p=sign_test_two_sided_p(wa, wb),
                        sep=sum(r['sep'] for r in rs), wa=wa, wb=wb))
    return out


def verdict_name(row):
    """The three-valued verdict, with the combined value spelled out."""
    if row['established'] and row['result']:
        return 'DIR+RESULT'
    if row['established']:
        return 'DIRECTION'
    if row['result']:
        return 'RESULT'
    return 'NULL'


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

    width = min(72, max(len(' / '.join(r['key'])) for r in rows))
    print(f"{'key':<{width}} {'n':>2} {'median':>8} {'diff':>7} "
          f"{'within':>7} {'r2r':>6} {'bar':>7} {'old':<7} {'new':<11} "
          f"{'magnitude':>17} {'sign':>9} {'p':>9} {'sep':>6}")
    for r in rows:
        k = ' / '.join(r['key'])
        mag = (f"{r['mag_lo']:>7.3f}-{r['mag_hi']:<9.3f}" if r['established']
               else f"{'--':>17}")
        print(f"{k[:width]:<{width}} {r['n']:>2} {r['median']:>8.4f} "
              f"{r['diff']:>7.3f} "
              f"{r['within']:>7.3f} {r['r2r']:>6.3f} {r['bar']:>7.3f} "
              f"{'RESULT' if r['result'] else 'NULL':<7} "
              f"{verdict_name(r):<11} {mag} "
              f"{r['wa']:>4}-{r['wb']:<4} {r['p']:>9.2g} {r['sep']:>2}/{r['n']:<3}")
    print("\n  median     the value to quote: the median of the per-run median ratios.")
    print("  diff       how many times apart the arms are, max(median, 1/median).")
    print("  within     median per-run swing of the per-round ratio (max/min).")
    print("  r2r        run-to-run scatter: max/min of the per-run medians.")
    print("  bar        the larger of within and r2r -- measure_util.hpp's rule.")
    print("  old        the two-valued verdict, on these same rounds.")
    print("  new        DIRECTION when no round crossed 1.00x, RESULT when diff > bar,")
    print("             DIR+RESULT when both, NULL when neither. Both are printed")
    print("             because they are different statements about the same rounds.")
    print("  magnitude  min..max per-round factor over every round of every run, in")
    print("             the established direction. That range is what a DIRECTION row")
    print("             is quoted as; it bounds the SIZE of the win, not its sign.")
    print("  sign       paired rounds favouring arm A - arm B, summed over runs.")
    print("  p          exact two-sided sign-test p over those rounds. REPORTED, and")
    print("             gated on by nothing: it is how two unanimous rounds are told")
    print("             apart from a hundred without a round-count bar being invented.")
    print("  sep        runs whose arms had DISJOINT ranges. A fact, not the verdict.")
    ties = [r for r in rows if r['tied'] and not r['established']
            and (r['wa'] == 0 or r['wb'] == 0)]
    if ties:
        print("\n  TIES held these rows back from DIRECTION -- every USABLE round fell")
        print("  one way, but some rounds favoured neither, and the sentence the")
        print("  verdict licenses is \"faster in N of N rounds\":")
        for r in ties:
            print(f"    {' / '.join(r['key'])[:width]}  {r['tied']} of {r['rounds']} tied")
    return 0


if __name__ == '__main__':
    sys.exit(main())
