#!/usr/bin/env python3
"""Aggregate N launches of a host benchmark and report what the harness cannot.

WHY THIS EXISTS.  benchmark/measure_util.hpp states the rule and, in the same
breath, the limit of what it can measure:

    A difference smaller than the spread is a null result, and a null result is
    a result.  The spread bounds WITHIN-run noise only; run-to-run scatter is a
    separate and sometimes larger number, and an entry that calls a difference
    real should clear the larger of the two.

One process can only ever print the first half.  Everything a launch pays once
-- where the allocator landed, which neighbour the scheduler put on the sibling
core, what the clock was doing when calibrate() chose the batch size -- is
constant inside that process, so it moves none of the batches it times and is
invisible to the spread it prints.  It moves between processes.

That gap is not a rounding correction on this hardware.  On the x86-64 desktop
under WSL2 goodFeaturesToTrack prints a ~5% within-run spread, and the same
ratio scatters 57% over thirty launches: the honest bound is eleven times the
printed one.  The Pi 4 with the governor locked is the other end -- 0.05-0.14%
within-run, 0.41% across launches -- which is what makes it the reference
device.  Until this script, every x86 figure under docs/reports/logs/ was one
draw from a distribution nobody had characterised, because all 24 of those logs
are single launches.  Not wrong; unexamined, which CLAUDE.md's "commit the
benchmark" rule does not allow to stand.

scripts/run_launches.sh produces the input.  This reads it.

WHAT IT READS.  The benchmarks print prose, not a machine-readable line, so the
parser recognises the two shapes the repository actually emits:

  * a fixed-width table whose header names a timing column -- `ns/pixel`,
    `ns/frame`, `us/call`, `ns/dst px` -- with the row's label in the columns
    left of it.  Numbered t1, t2, ... in the order they appear within a launch,
    so the seven identically-headed tables in the morphology log stay seven
    groups of rows here rather than collapsing into one.
  * `[BENCH] <label> ... <value> ns/px ...` lines, and the `- Avg: <value> ms`
    form from bench_util.hpp.  Numbered b1, b2, ... per contiguous run, which
    is what keeps the logic log's three frame sizes apart.
  * a bare `<label>  <value> ms` line with no table around it, which is how the
    LK head-to-head and the feature-tracking profile print their stages.
    Numbered i1, i2, ...

A row seen twice inside one group gets a `#2` suffix rather than overwriting the
first -- a silently discarded measurement is a wrong number that looks right.  A
file that yields nothing is named on stderr; six of the committed logs print
their timings in shapes none of the above covers, and a reader has to be told
that rather than left to read a short table as the whole story.

A launch begins at a line like `### run 7`; the committed logs' `### vfy x86
run 7 (FULL)` matches the same pattern.  A file with no such marker is treated
as one launch, so a directory of per-launch files aggregates too.

WHAT IT COMPUTES, per row, over the launches that carry it:

  median    the median of the per-launch medians -- the value to quote.  Each
            per-launch number is already a median over batches, which is the
            within-run half; this is the between-launch half of the same idea.
  min max   the extremes across launches.
  r2r       max / min of the per-launch medians.  The run-to-run scatter, as a
            factor, the same definition scripts/aggregate_cuda_runs.py uses --
            one definition of this quantity in the repository, not two.
  scatter   (max - min) / median across launches.  The SAME arithmetic
            Timing::spreadPct() does inside a process, so it can be read
            directly against the `within` column beside it.
  within    the median of the per-launch spread column, when the table has one.
            This is what a single launch reports about itself.  The two columns
            side by side are the whole point of the script.
  ci95      percentile bootstrap interval on the median, over launches.
  +/-       half that interval's width, as a percentage of the median.
  minres    THE SMALLEST DIFFERENCE THIS MANY LAUNCHES CAN RESOLVE ON THIS ROW:
            max(hi/median, median/lo).  A change smaller than this factor moves
            the median to somewhere the current interval already covers, so
            these launches cannot tell it from no change at all.  It is derived
            from the row's own scatter and is not a threshold anyone chose --
            `--resolve` compares it against a difference the CALLER states,
            because how much is worth having is a per-case judgement and
            CLAUDE.md forbids inventing a project-wide one here.

RATIOS ARE PAIRED PER LAUNCH BY DEFAULT.  `--ratio "A/B"` divides A by B inside
each launch and then takes the median of those ratios, which is what the
goodFeaturesToTrack figure in docs/reports/logs/goodfeatures-x86_64.log is.  The
ratio of the two rows' medians is printed beside it, over the same launches,
because the two answer slightly different questions and diverge exactly when the
per-launch ratios are skewed -- on that row they are, by two slow-OpenCV
launches, and the two intervals come out [1.1120, 1.1537] and [1.1150, 1.1589]
off the same thirty launches.  Quoting one while having computed the other is
how a reproduction silently fails to reproduce; it cost an afternoon here.

Pairing is reported, NOT assumed to help.  It cancels a disturbance that hits
both arms of a launch together, and on this host the goodFeaturesToTrack ratio
scatters 57% while its slower arm alone scatters 44% -- so there the pairing
buys nothing, and the `scatter` column for the ratio row says so in the numbers
rather than in a hope.

THE LADDER (`--ladder`) ANSWERS "HOW MANY LAUNCHES".  For each k it resamples k
launches with replacement from the ones observed, takes the median, and reports
the band those k-launch experiments would have landed in.  At k = n this is the
ordinary bootstrap interval; below it, it is what a shorter sweep would have
told you.  At k = 1 the band is the whole observed spread, which is the finding
that started this: one launch resolves nothing, and cannot report that it does
not.

USAGE
    scripts/aggregate_launches.py <log-or-dir> [...] [options]

    --column NAME     which header column is the timing (default: the first
                      one that looks like a time)
    --ratio "A/B"     also report the per-launch ratio of row A to row B
    --resolve D       is a difference of D resolvable here?  "1.05" or "5%"
    --ladder          launch-count ladder for every reported row
    -k, --key REGEX   only rows whose key matches
    --resamples N     bootstrap resamples (default 10000)
    --seed N          bootstrap seed (default 12345); printed, so a figure can
                      be reproduced exactly rather than approximately
"""

import argparse
import os
import random
import re
import sys
from statistics import median

# A header cell that names a time per something: ns/pixel, us/call, ms/frame.
# GB/s is a rate and is deliberately not matched -- it is not what a ratio of
# durations is taken on -- but --column will still select it by name.
TIMING_COL = re.compile(r'^(?:ns|us|ms|s)/[A-Za-z][A-Za-z /]*$')
SECTION = re.compile(r'^###\s*(.*)$')
RUN_MARK = re.compile(r'\b(?:run|repeat|launch)\s+(\d+)\b', re.I)
BENCH_NS = re.compile(r'\[BENCH\]\s*(.*?)\s{2,}(\d+(?:\.\d+)?)\s+'
                      r'(?:ns|us|ms|s)/\S+')
BENCH_AVG = re.compile(r'\[BENCH\]\s*(.*?)\s+-\s+Avg:\s*([\d.eE+-]+)\s+ms')
RULE = re.compile(r'^[-=\s]+$')
# `<label>  <value> ms` with no table around it: the LK head-to-head and the
# feature-tracking profile print their stage times that way. Two or more spaces
# before the number and a TIME unit right after it are what keep this off
# `binCV :   436704 B` and the rest of the prose.
INLINE = re.compile(r'^\s{0,6}(?P<label>\S.*?)\s{2,}'
                    r'(?P<v>\d+(?:\.\d+)?)\s*'
                    r'(?:ns|us|ms|s)\b(?:/\S+)?')


def _number(cell):
    """The cell as a float, or None. Commas group; a trailing % or x is a unit."""
    s = cell.strip().rstrip('%x').replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _cells(line):
    """Split a fixed-width row. Two or more spaces separate columns, so a label
    like `binCV streaming` or a header like `vs OpenCV` stays one cell."""
    return re.split(r'\s{2,}', line.strip())


def split_launches(text):
    """[(label, lines)] for one file, one entry per LAUNCH.

    `###` opens a section; a section is a launch when its title names a run,
    repeat or launch number. Everything else -- the goodfeatures log's twelve
    `### pair 3 first=old` blocks, its `### AGGREGATE` summary -- is dropped
    rather than folded into whichever launch preceded it, which is what made the
    pair tables show up as twenty-four extra tables inside run 30.

    A file with no `###` at all is one launch, so a directory holding one file
    per launch aggregates the same way.
    """
    lines = text.split('\n')
    # Everything before the first `###` is a launch too. A log with no section
    # markers at all -- lk_headtohead-x86_64.log is one -- is entirely preamble,
    # and an earlier version of this function dropped every such file on the
    # floor while reporting "no rows parsed", which reads like a parser failure
    # rather than the bookkeeping slip it was.
    out, cur, label = [], [], '0'
    for line in lines:
        m = SECTION.match(line)
        if m:
            if label is not None:
                out.append((label, cur))
            r = RUN_MARK.search(m.group(1))
            label, cur = (r.group(1) if r else None), []
        else:
            cur.append(line)
    if label is not None:
        out.append((label, cur))
    return out


def parse_launch(lines, column):
    """One launch -> {(table, row): (value, within_pct_or_None)}, plus table legend.

    A header is a line with several cells, at least one of which names a time and
    none of which is a bare number -- which is what separates it from the body
    rows underneath it.
    """
    rows, legend = {}, {}
    table = bench_block = inline_block = 0
    prev_was_inline = False
    header = None
    vcol = scol = None
    caption = ''
    prev_was_bench = False

    def put(group, label, value, within):
        """Record a row, never overwriting one.

        The logic benchmark prints the same [BENCH] labels once per frame size,
        so a plain assignment kept the last size and silently discarded the other
        two -- a wrong number that looks like a right one. A repeat inside one
        group gets a `#2` suffix instead, which at worst is ugly and at best says
        the log has a shape the grouping did not anticipate.
        """
        key, i = (group, label), 1
        while key in rows:
            i += 1
            key = (group, f'{label} #{i}')
        rows[key] = (value, within)

    for line in lines:
        if not line.strip():
            # A blank line closes both a table and a run of inline lines, so the
            # groups below line up with the blocks a reader sees in the log.
            header = None
            prev_was_inline = False
            continue

        m = BENCH_NS.search(line) or BENCH_AVG.search(line)
        if m:
            if not prev_was_bench:
                bench_block += 1
                legend[f'b{bench_block}'] = (caption or '[BENCH] lines')
            put(f'b{bench_block}', re.sub(r'\s+', ' ', m.group(1)).strip(),
                float(m.group(2)), None)
            header, prev_was_bench, prev_was_inline = None, True, False
            continue
        prev_was_bench = False

        cs = _cells(line)
        if header is None:
            m = (None if line.lstrip().startswith(('#', '=', '-'))
                 else INLINE.match(line))
            if m and not any(TIMING_COL.match(c) for c in cs):
                if not prev_was_inline:
                    inline_block += 1
                    legend[f'i{inline_block}'] = (caption or 'inline <value> <unit>')
                put(f'i{inline_block}', re.sub(r'\s+', ' ', m.group('label')).strip(),
                    float(m.group('v')), None)
                prev_was_inline = True
                continue
            prev_was_inline = False
            if (len(cs) >= 2 and any(TIMING_COL.match(c) for c in cs)
                    and not any(_number(c) is not None for c in cs)):
                # Pick the timing column: the caller's by name, else the first
                # that looks like a time.
                if column:
                    want = [i for i, c in enumerate(cs) if c.lower() == column.lower()]
                    # A table without the named column is CONSUMED and dropped,
                    # not left for the inline matcher to pick rows out of, and it
                    # still takes its ordinal -- so t3 means the same table
                    # whether or not --column was passed.
                    vcol = want[0] if want else None
                else:
                    vcol = next(i for i, c in enumerate(cs) if TIMING_COL.match(c))
                scol = next((i for i, c in enumerate(cs) if c.lower() == 'spread'), None)
                table += 1
                header = cs
                legend[f't{table}'] = (caption + ' | ' if caption else '') + '  '.join(cs)
            else:
                caption = re.sub(r'\s+', ' ', line.strip())[:58]
            continue

        if RULE.match(line):
            continue
        if len(cs) != len(header):
            header = None
            caption = re.sub(r'\s+', ' ', line.strip())[:58]
            continue
        if vcol is None:
            continue
        v = _number(cs[vcol])
        if v is None:
            header = None
            continue
        w = _number(cs[scol]) if scol is not None else None
        # The label is the LEADING RUN of non-numeric columns. The logic
        # benchmark splits its label over two ("OP" and "IMPLEMENTATION"), and
        # taking only the first would collapse three implementations of
        # bitwiseAnd into one name; taking everything left of the timing column
        # would glue a ns/frame figure onto the label when --column selects the
        # ns/pixel column beside it.
        lead = []
        for c in cs[:max(vcol, 1)]:
            if _number(c) is not None:
                break
            lead.append(c)
        label = ' '.join(lead) if lead else cs[0]
        put(f't{table}', label, v, w)
        prev_was_inline = False
    return rows, legend


def collect(paths, column):
    """-> (key -> [per-launch value]), (key -> [per-launch within%]), legend, n."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, n) for n in sorted(os.listdir(p))
                      if os.path.isfile(os.path.join(p, n))]
        else:
            files.append(p)
    vals, withins, legend, launches, silent = {}, {}, {}, 0, []
    for path in files:
        with open(path, errors='replace') as fh:
            text = fh.read()
        before = launches
        for _, lines in split_launches(text):
            rows, leg = parse_launch(lines, column)
            if not rows:
                continue
            launches += 1
            legend.update(leg)
            for k, (v, w) in rows.items():
                vals.setdefault(k, []).append(v)
                if w is not None:
                    withins.setdefault(k, []).append(w)
        if launches == before:
            silent.append(path)
    return vals, withins, legend, launches, silent


def boot_band(samples, k, resamples, seed):
    """Percentile band of the median of k launches drawn with replacement.

    At k == len(samples) this is the ordinary percentile bootstrap interval on
    the median. Below it, it is the band a k-launch sweep would have landed in,
    estimated from the launches actually observed -- which is how the ladder
    answers "how many launches" without assuming the scatter is normal or that
    it shrinks like the square root of anything.
    """
    rng = random.Random(seed)
    meds = sorted(median(rng.choices(samples, k=k)) for _ in range(resamples))
    return meds[int(0.025 * resamples)], meds[int(0.975 * resamples)]


def stats(samples, withins, resamples, seed):
    med = median(samples)
    out = dict(n=len(samples), median=med, min=min(samples), max=max(samples),
               r2r=max(samples) / min(samples) if min(samples) > 0 else float('inf'),
               scatter=(max(samples) - min(samples)) / med * 100.0 if med else 0.0,
               within=median(withins) if withins else None,
               lo=None, hi=None, half=None, minres=None)
    if len(samples) < 2:
        # A SINGLE LAUNCH HAS NO RUN-TO-RUN INFORMATION, and the bootstrap of one
        # sample would hand back a zero-width interval and a minres of 1.000x --
        # a claim that this launch resolves any difference whatsoever, which is
        # precisely the error the script exists to stop. Left blank on purpose.
        return out
    out['lo'], out['hi'] = boot_band(samples, len(samples), resamples, seed)
    out['half'] = (out['hi'] - out['lo']) / 2.0 / med * 100.0 if med else 0.0
    out['minres'] = (max(out['hi'] / med, med / out['lo'])
                     if med and out['lo'] else float('inf'))
    return out


def parse_diff(s):
    """"1.05" or "5%" -> 1.05. A difference is a FACTOR here, never a percentage
    of one arm: see aggregate_cuda_runs.py for why that distinction is not
    presentation."""
    s = s.strip()
    if s.endswith('%'):
        return 1.0 + float(s[:-1]) / 100.0
    return float(s)


LADDER = (1, 2, 3, 5, 10, 15, 20, 30, 50)


def print_ladder(name, samples, resamples, seed, want=None):
    n = len(samples)
    med = median(samples)
    ks = sorted({k for k in LADDER if k <= n} | {n})
    print(f"\n  launch ladder for {name}  (median of the {n} observed launches"
          f" = {med:.4f})")
    print(f"    {'launches':>8} {'2.5%':>9} {'97.5%':>9} {'+/-':>8} "
          f"{'resolves':>10}" + ("   stated" if want else ""))
    for k in ks:
        lo, hi = boot_band(samples, k, resamples, seed)
        half = (hi - lo) / 2.0 / med * 100.0
        mr = max(hi / med, med / lo) if lo else float('inf')
        mark = ''
        if want:
            mark = '   YES' if mr <= want else '   no'
        print(f"    {k:>8} {lo:>9.4f} {hi:>9.4f} {half:>7.2f}% "
              f"{mr:>9.3f}x{mark}")
    if 1 in ks:
        print("    k=1 is the state of every committed x86 log: the band is the whole")
        print("    observed spread, and a single launch cannot report it.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='launch logs, or directories of them')
    ap.add_argument('--column', default='', help='timing column to read')
    ap.add_argument('--ratio', default='', help='"numerator row/denominator row"')
    ap.add_argument('--resolve', default='', help='a difference to test: 1.05 or 5%%')
    ap.add_argument('--ladder', action='store_true', help='launch-count ladder')
    ap.add_argument('-k', '--key', default='', help='regex over the row key')
    ap.add_argument('--resamples', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args(argv)

    vals, withins, legend, launches, silent = collect(args.paths, args.column)
    # A file that yields nothing is NAMED. Silence from a parser reads as "this
    # log has no rows", and several of the committed logs print their timings in
    # shapes nothing here recognises -- a fact a reader has to be told rather
    # than left to infer from a short table.
    if silent:
        print(f"no rows parsed from {len(silent)} file(s) -- their timings are"
              f" printed in a shape\nthis parser does not recognise, which is not"
              f" the same as their having none:", file=sys.stderr)
        for path in silent:
            print(f"    {path}", file=sys.stderr)
    if not vals:
        print('nothing to aggregate (try --column against the log header)',
              file=sys.stderr)
        return 1
    want = parse_diff(args.resolve) if args.resolve else None

    rows = []
    for key, samples in sorted(vals.items()):
        name = ' / '.join(key)
        if args.key and not re.search(args.key, name, re.I):
            continue
        rows.append((name, key, stats(samples, withins.get(key, []),
                                      args.resamples, args.seed)))
    if not rows and not args.ratio:
        print('no rows matched', file=sys.stderr)
        return 1

    ragged = sorted({s['n'] for _, _, s in rows})
    print(f"{launches} launches parsed from {len(args.paths)} path(s); "
          f"bootstrap {args.resamples} resamples, seed {args.seed}")
    if len(ragged) > 1:
        print(f"  NOTE: rows appear in different numbers of launches {ragged} -- a"
              f" benchmark whose\n  table shape varies between launches is parsed"
              f" per launch, so check the legend.")

    w = min(46, max([len(r[0]) for r in rows] + [3]))
    print()
    if rows:
        print(f"{'row':<{w}} {'n':>3} {'median':>10} {'min':>10} {'max':>10} "
              f"{'r2r':>6} {'scatter':>8} {'within':>7} "
              f"{'ci95 of the median':>22} {'+/-':>7} {'minres':>8}")
    for name, _, s in rows:
        within = f"{s['within']:>6.2f}%" if s['within'] is not None else f"{'--':>7}"
        if s['lo'] is None:
            tail = f"  {'ONE LAUNCH -- no interval':>22} {'--':>7} {'--':>8}"
        else:
            tail = (f"  [{s['lo']:>8.4f}, {s['hi']:>8.4f}] {s['half']:>6.2f}% "
                    f"{s['minres']:>7.3f}x")
        print(f"{name[:w]:<{w}} {s['n']:>3} {s['median']:>10.4f} {s['min']:>10.4f} "
              f"{s['max']:>10.4f} {s['r2r']:>6.3f} {s['scatter']:>7.1f}% {within}"
              f"{tail}")
    if any(s['n'] < 2 for _, _, s in rows):
        print("\n  A ROW SEEN IN ONE LAUNCH HAS NO INTERVAL. The columns are blank"
              " because a\n  single process carries no run-to-run information, not"
              " because it has none\n  to carry -- which is the state of every"
              " committed x86 log under docs/reports/logs/.")

    if args.ratio:
        num, den = [p.strip() for p in args.ratio.split('/', 1)]

        def find(spec):
            """A row by label, or by `t2:label` when the label is in more than
            one table. Ambiguity is an error rather than a first-match, because
            picking a table silently is how a ratio ends up comparing the 640x480
            arm against the 8192x4096 one."""
            if ':' in spec:
                t, lbl = spec.split(':', 1)
                hits = [k for k in vals if k[0] == t.strip() and k[1] == lbl.strip()]
            else:
                hits = [k for k in vals if k[1] == spec]
            if not hits:
                print(f"\nratio: no row labelled {spec!r}", file=sys.stderr)
                return None
            if len(hits) > 1:
                print(f"\nratio: {spec!r} is in {len(hits)} tables "
                      f"({', '.join(sorted(h[0] for h in hits))}); "
                      f"qualify it as e.g. {sorted(hits)[0][0]}:{spec}", file=sys.stderr)
                return None
            return hits[0]

        kn, kd = find(num), find(den)
        if kn is None or kd is None:
            return 1
        a, b = vals[kn], vals[kd]
        if len(a) != len(b):
            print("\nratio: the two rows were not seen in the same launches",
                  file=sys.stderr)
            return 1
        pairs = [x / y for x, y in zip(a, b)]
        s = stats(pairs, [], args.resamples, args.seed)
        # The unpaired estimator, over the same launches: resample launches
        # jointly, then divide the two medians.
        rng = random.Random(args.seed)
        n = len(a)
        outs = []
        for _ in range(args.resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            outs.append(median([a[i] for i in idx]) / median([b[i] for i in idx]))
        outs.sort()
        ulo, uhi = outs[int(0.025 * args.resamples)], outs[int(0.975 * args.resamples)]
        upoint = median(a) / median(b)

        print(f"\nRATIO  {num}  /  {den}   over {n} launches")
        print(f"  paired per launch (the published estimator)")
        print(f"    median {s['median']:.4f}   min {s['min']:.4f}   max {s['max']:.4f}"
              f"   scatter {s['scatter']:.1f}%")
        if s['lo'] is None:
            print("    ONE LAUNCH: no interval, and no resolvable difference to")
            print("    report. This is a number, not yet a measurement.")
            return 0
        print(f"    bootstrap 95% CI of the median: "
              f"[{s['lo']:.4f}, {s['hi']:.4f}]   +/- {s['half']:.2f}%")
        print(f"    launches with the ratio above 1.00: "
              f"{sum(1 for p in pairs if p > 1.0)} of {n}")
        print(f"    smallest resolvable difference at {n} launches: "
              f"{s['minres']:.3f}x")
        print(f"  ratio of the two medians, same launches")
        print(f"    point {upoint:.4f}   bootstrap 95% CI: [{ulo:.4f}, {uhi:.4f}]")
        # Always printed, never conditional: the two are different estimators
        # of different things, and a quoted interval that does not say which one
        # produced it cannot be reproduced. The gaps are in units of the median,
        # so they read against the +/- above them.
        print(f"    the two intervals differ by {abs(ulo - s['lo']) / s['median'] * 100:.2f}%"
              f" and {abs(uhi - s['hi']) / s['median'] * 100:.2f}% of the median at the"
              f" two ends.\n    Say which estimator a quoted interval came from. The"
              f" paired one is what the\n    committed logs publish.")
        if want:
            print(f"  a difference of {want:.4f}x is "
                  f"{'RESOLVABLE' if s['minres'] <= want else 'NOT resolvable'}"
                  f" at {n} launches on this row")
        if args.ladder:
            print_ladder(f"{num} / {den}", pairs, args.resamples, args.seed, want)

    if want and not args.ratio:
        print(f"\nA difference of {want:.4f}x, against each row's own interval:")
        for name, _, s in rows:
            if s['minres'] is None:
                print(f"  {name[:w]:<{w}}  UNANSWERABLE from one launch")
                continue
            print(f"  {name[:w]:<{w}}  {'RESOLVABLE' if s['minres'] <= want else 'NOT resolvable'}"
                  f"   (these launches resolve {s['minres']:.3f}x)")

    if args.ladder and not args.ratio:
        for name, key, _ in rows:
            print_ladder(name, vals[key], args.resamples, args.seed, want)

    print("\n  median   median of the per-launch medians -- the value to quote.")
    print("  r2r      run-to-run scatter as a factor, max/min of those medians.")
    print("  scatter  (max-min)/median ACROSS launches: Timing::spreadPct()'s own")
    print("           arithmetic, so it reads directly against the next column.")
    print("  within   median of the per-launch spread column -- what one launch")
    print("           reports about itself, and all it can report.")
    print("  minres   smallest difference these launches resolve on this row,")
    print("           max(hi/median, median/lo). Derived from the row's scatter;")
    print("           whether it is small ENOUGH is a per-case call, so state the")
    print("           difference with --resolve rather than reading a bar into it.")
    if legend:
        print("\n  tables in a launch, in order:")
        for t, h in sorted(legend.items()):
            print(f"    {t:<8} {h}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
