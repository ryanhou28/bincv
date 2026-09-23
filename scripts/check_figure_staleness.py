#!/usr/bin/env python3
"""check_figure_staleness.py -- a published figure is stale when the code under it moved.

WHY THIS EXISTS.  `verify.sh` gates whether a kernel is CORRECT.  Nothing gated
whether a published figure was still TRUE, and the two come apart exactly when an
optimization preserves every output bit -- which is the change this project makes
most often.  Measured cost of that gap: a response-sweep optimization made
`goodFeaturesToTrack` 18% faster on both architectures and left the reports
publishing a LOSS the library did not have, for three weeks.  `features.md` even
carried a note saying the rows predated the optimization.  The note was not
enough, and nothing mechanical could see the problem.

WHAT IT CHECKS.  `scripts/run_launches.sh` already stamps every log it writes with
the commit it was taken at.  So for each committed log this asks one question:

    does the code this benchmark MEASURES still say what it said then?

The mapping from a log to that code is the part the issue called hard, and it is
the C preprocessor's own: a log names its benchmark binary, the binary has a
source file, and that source `#include`s a transitive set of first-party headers.
Every file in that set is then COMPARED, at the log's commit against the working
tree.  If any of them differs, every figure quoted from that log is a figure
about code that no longer exists.

IT COMPARES CONTENT, NOT HISTORY, and that is not a stylistic choice.  Asking
"what has `git log <stamp>..HEAD` touched" is wrong the moment a branch is
SQUASH-merged: the squash commit is not a descendant of the stamp, so the whole
branch reads as change-since, and every log taken on that branch reports stale
against the code it was actually taken on.  This repository squash-merges, so
that is not a hypothetical -- it was the first thing this gate got wrong.
Comparing the files themselves is immune to squashes, rebases and cherry-picks
alike, because it never asks how the code got there.

COMMENTS DO NOT COUNT, AND THAT IS THE DIFFERENCE BETWEEN A GATE AND AN ALARM.
Asked naively -- "has git touched any header this benchmark includes?" -- every
log in the repository is stale, because one comment edit to `core/error.hpp`
reaches all of them.  A gate that is always red is a gate that gets switched off.
So each commit is classified once by whether it changed CODE in a file: a hunk
whose added and removed lines are all comment or blank does not age a figure,
because it cannot have changed a number.  The test is lexical -- `//`, `/*`, `*`,
`*/`, and blank -- so a comment containing a brace is judged comment, and a code
line with a trailing comment is judged code.  It errs toward calling a change
code, which is the safe direction.

WHAT IT DOES NOT CHECK.  Whether a figure is RIGHT -- only whether the code under
it has moved since it was taken.  A report quoting a number that was never in any
log is invisible here, as is a figure written into a source comment; the first is
what `check_report_figures.py` covers and the second is a convention
(a commit named beside the figure) that no gate enforces.

THE BASELINE, AND WHY IT RECORDS FILES RATHER THAN LOG NAMES.  Most committed
logs are already stale, and failing on all of them would make this gate something
to switch off.  So the known-stale set lives in
`docs/reports/logs/expected-stale.txt` and only NEW staleness fails -- the same
shape as `tests/expected-checks.txt`.

A baseline of log NAMES would have been useless here, and it was tried: with every
mappable log already listed, editing a kernel changed nothing the gate would
report, because each log was already exempt.  A gate that cannot go red is the
thing it was built to replace.  So each entry records the SET OF FILES known to
have moved under that log, and a log fails when that set GROWS.  An already-stale
figure does not excuse the next change to the code beneath it.

A log leaves the file by being RE-TAKEN; `--update-baseline` rewrites it, and that
edit is meant to be read in review rather than waved through.

Exits non-zero when a log has gone stale that the baseline does not already name.

Usage: python3 scripts/check_figure_staleness.py [--update-baseline] [-v]
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'docs', 'reports', 'logs')
REPORTS = os.path.join(ROOT, 'docs', 'reports')
BASELINE = os.path.join(LOGS, 'expected-stale.txt')

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def git(*args):
    """git, from the repository root, stdout as text ('' on failure)."""
    try:
        out = subprocess.run(['git', '-C', ROOT] + list(args),
                             capture_output=True, text=True, check=False)
        return out.stdout if out.returncode == 0 else ''
    except OSError:
        return ''


def first_party_deps(source, _seen=None):
    """`source` plus every first-party header it reaches, as repo-relative paths.

    Resolution is the compiler's: a quoted include is tried against the including
    file's own directory first, then `include/`. Anything that resolves outside
    the repository -- a system or OpenCV header -- is not ours and is dropped,
    because a figure does not go stale when OpenCV does. That is a real limit and
    it is stated rather than hidden: an OpenCV upgrade moves denominators and
    nothing here will say so.
    """
    seen = _seen if _seen is not None else set()
    rel = os.path.relpath(source, ROOT)
    if rel in seen or not os.path.isfile(source):
        return seen
    seen.add(rel)
    try:
        text = open(source, encoding='utf-8', errors='replace').read()
    except OSError:
        return seen
    for inc in INCLUDE_RE.findall(text):
        for base in (os.path.dirname(source), os.path.join(ROOT, 'include')):
            cand = os.path.normpath(os.path.join(base, inc))
            if cand.startswith(ROOT) and os.path.isfile(cand):
                first_party_deps(cand, seen)
                break
    return seen


def strip_comments(text):
    """`text` with C++ comments and blank lines removed, for comparison only.

    A small state machine rather than a regex, because `//` inside a string
    literal is not a comment and this has to agree with the compiler often enough
    to be trusted. String and character literals are tracked for that reason
    alone; nothing else about the language is modelled.
    """
    out, i, n = [], 0, len(text)
    state = 'code'  # code | line_comment | block_comment | string | char
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ''
        if state == 'code':
            if c == '/' and nxt == '/':
                state, i = 'line_comment', i + 2
                continue
            if c == '/' and nxt == '*':
                state, i = 'block_comment', i + 2
                continue
            if c == '"':
                state = 'string'
            elif c == "'":
                state = 'char'
            out.append(c)
        elif state == 'line_comment':
            if c == '\n':
                state = 'code'
                out.append(c)
        elif state == 'block_comment':
            if c == '*' and nxt == '/':
                state, i = 'code', i + 2
                continue
            if c == '\n':
                out.append(c)  # keep line structure so a diff stays readable
        elif state in ('string', 'char'):
            out.append(c)
            if c == '\\':
                if i + 1 < n:
                    out.append(nxt)
                i += 2
                continue
            if (state == 'string' and c == '"') or (state == 'char' and c == "'"):
                state = 'code'
        i += 1
    return '\n'.join(ln.strip() for ln in ''.join(out).splitlines() if ln.strip())


def blob_at(rev, path, cache={}):
    """`path` as of `rev`, or None when it did not exist there."""
    key = (rev, path)
    if key not in cache:
        out = subprocess.run(['git', '-C', ROOT, 'show', '%s:%s' % (rev, path)],
                             capture_output=True, text=True, check=False)
        cache[key] = out.stdout if out.returncode == 0 else None
    return cache[key]


def code_differs(rev, path):
    """Whether `path` differs in CODE between `rev` and the working tree."""
    then = blob_at(rev, path)
    now_path = os.path.join(ROOT, path)
    now = None
    if os.path.isfile(now_path):
        try:
            now = open(now_path, encoding='utf-8', errors='replace').read()
        except OSError:
            now = None
    if then is None and now is None:
        return False
    if then is None or now is None:
        return True          # added or deleted since -- a real change either way
    if then == now:
        return False         # identical bytes, the common case, no stripping needed
    return strip_comments(then) != strip_comments(now)


def benchmark_source(binary):
    """The source file behind a benchmark binary, or None.

    Benchmarks are one translation unit named after the binary, in benchmark/ or
    tests/. A binary this cannot place is reported rather than assumed.
    """
    stem = os.path.basename(binary)
    for d in ('benchmark', 'tests', 'examples'):
        for ext in ('.cpp', '.cc'):
            cand = os.path.join(ROOT, d, stem + ext)
            if os.path.isfile(cand):
                return cand
    return None


def read_header(path):
    """The `# key: value` preamble run_launches.sh writes, as a dict."""
    fields = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not line.startswith('#'):
                    if line.strip() and not line.startswith('#'):
                        break
                    continue
                m = re.match(r'#\s*([a-z][a-z ]*?)\s*:\s*(.*)$', line.strip())
                if m:
                    fields.setdefault(m.group(1).strip(), m.group(2).strip())
    except OSError:
        pass
    return fields


def reports_linking(logname):
    """Which reports link this log, so a failure names what a reader would see."""
    hits = []
    for dirpath, _dirs, files in os.walk(REPORTS):
        for f in files:
            if not f.endswith('.md'):
                continue
            p = os.path.join(dirpath, f)
            try:
                if logname in open(p, encoding='utf-8', errors='replace').read():
                    hits.append(os.path.relpath(p, ROOT))
            except OSError:
                pass
    return sorted(hits)


def self_check():
    """Prove the comment/code distinction still works, before trusting a run.

    This gate's entire value is that it stays quiet for a comment edit and fires
    for a code one. If `strip_comments` regressed, it would go quiet for BOTH and
    report a clean repository -- the failure mode that looks like success, which
    is the one `verify.sh` grew its own self-check for. Four cases, on strings, so
    it costs nothing and runs every time.
    """
    base = 'int f() {\n  return 1;  // one\n}\n'
    cases = [
        ('comment appended', base + '// a note\n', False),
        ('block comment appended', base + '/* a\n   note */\n', False),
        ('code appended', base + 'int g() { return 2; }\n', True),
        ('a literal containing //', base, False),
    ]
    for label, other, want_differ in cases:
        got = strip_comments(base) != strip_comments(other)
        if got != want_differ:
            print('SELF-CHECK FAILED: %s -- expected differ=%s, got %s'
                  % (label, want_differ, got), file=sys.stderr)
            return False
    # A `//` inside a string is not a comment, and treating it as one would make
    # two different lines of code compare equal.
    if strip_comments('const char* u = "a//b";') == strip_comments('const char* u = "a";'):
        print('SELF-CHECK FAILED: a // inside a string literal was treated as a comment',
              file=sys.stderr)
        return False
    return True


def load_baseline():
    """{log name: (stamp, {files known stale})} from the baseline file."""
    if not os.path.isfile(BASELINE):
        return {}
    out = {}
    for line in open(BASELINE, encoding='utf-8'):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split('\t')
        name = parts[0].strip()
        stamp = parts[1].strip() if len(parts) > 1 else ''
        files = set(f.strip() for f in parts[2].split(',') if f.strip()) if len(parts) > 2 else set()
        out[name] = (stamp, files)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--update-baseline', action='store_true',
                    help='rewrite expected-stale.txt from what is stale now')
    ap.add_argument('--note', default='',
                    help='a line recorded in the baseline saying WHY this batch is accepted')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='list the fresh and unmappable logs too')
    args = ap.parse_args()

    if not self_check():
        return 2

    if not git('rev-parse', '--git-dir'):
        print('not a git repository -- cannot tell when anything was taken')
        return 0

    fresh, stale, unmappable = [], {}, {}

    for name in sorted(os.listdir(LOGS)):
        if not name.endswith('.log'):
            continue
        path = os.path.join(LOGS, name)
        head = read_header(path)

        commit = head.get('commit', '')
        if not commit:
            unmappable[name] = 'no commit stamp -- predates run_launches.sh recording one'
            continue
        if '(dirty)' in commit:
            unmappable[name] = 'taken from a modified tree, so it names no commit'
            continue
        commit = commit.split()[0]
        if not git('cat-file', '-e', commit + '^{commit}') and not git('rev-parse', '--verify',
                                                                       '--quiet', commit):
            unmappable[name] = 'commit %s is not in this history' % commit
            continue

        binary = head.get('benchmark', '').split()[0] if head.get('benchmark') else ''
        if not binary:
            unmappable[name] = 'no benchmark line -- cannot tell what it measured'
            continue
        source = benchmark_source(binary)
        if source is None:
            unmappable[name] = 'no source found for %s' % os.path.basename(binary)
            continue

        deps = sorted(first_party_deps(source))
        moved = [d for d in deps if code_differs(commit, d)]
        if moved:
            stale[name] = (commit, moved)
        else:
            fresh.append((name, commit, len(deps)))

    if args.update_baseline:
        with open(BASELINE, 'w', encoding='utf-8') as fh:
            fh.write('# Logs whose measured code has moved since they were taken.\n'
                     '# Written by scripts/check_figure_staleness.py --update-baseline.\n'
                     '# A log leaves this list by being RE-TAKEN, not by being edited.\n'
                     '# Adding a line here is a decision to publish a figure that is known\n'
                     '# not to describe the current code, and belongs in review.\n')
            if args.note:
                fh.write('#\n')
                for chunk in args.note.split('|'):
                    chunk = chunk.strip()
                    fh.write(('# %s\n' % chunk) if chunk else '#\n')
            fh.write('#\n')
            for name in sorted(stale):
                commit, moved = stale[name]
                fh.write('%s\t%s\t%s\n' % (name, commit, ','.join(moved)))
        print('wrote %s (%d stale)' % (os.path.relpath(BASELINE, ROOT), len(stale)))
        return 0

    baseline = load_baseline()
    newly = []
    for name, (commit, moved) in sorted(stale.items()):
        known_stamp, known_files = baseline.get(name, ('', set()))
        if name not in baseline:
            newly.append((name, commit, moved, 'not recorded'))
        elif known_stamp and known_stamp != commit:
            newly.append((name, commit, moved, 're-taken at %s; baseline names %s'
                          % (commit, known_stamp)))
        else:
            grew = sorted(set(moved) - known_files)
            if grew:
                newly.append((name, commit, grew, 'newly changed since the baseline'))
    recovered = sorted(set(baseline) - set(stale) - set(unmappable))

    print('figure staleness: %d fresh, %d stale, %d unmappable, of %d logs'
          % (len(fresh), len(stale), len(unmappable), len(fresh) + len(stale) + len(unmappable)))

    if args.verbose:
        for name, commit, n in fresh:
            print('  FRESH      %-52s %s, %d files' % (name, commit, n))
        for name in sorted(unmappable):
            print('  UNMAPPABLE %-52s %s' % (name, unmappable[name]))

    for name, commit, moved, why in newly:
        print('\n  STALE (new)  %s' % name)
        print('    taken at %s -- %s:' % (commit, why))
        for t in moved[:6]:
            print('      %s' % t)
        if len(moved) > 6:
            print('      ... and %d more' % (len(moved) - 6))
        cited = reports_linking(name)
        if cited:
            print('    quoted by: %s' % ', '.join(cited))

    if recovered:
        print('\n  no longer stale (re-taken) -- drop from the baseline:')
        for name in recovered:
            print('      %s' % name)

    if newly:
        print('\n  %d log(s) went stale. Re-take them, or record the decision to publish a\n'
              "  figure that does not describe current code:\n"
              '      python3 scripts/check_figure_staleness.py --update-baseline'
              % len(newly))
        return 1

    print('  no NEW staleness (%d already recorded in %s)'
          % (len(baseline), os.path.relpath(BASELINE, ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
