#!/usr/bin/env python3
"""check_links.py -- every relative link in the repository resolves to a real file.

WHY THIS EXISTS. The top-level design documents moved into docs/ on 2026-08-31, which
was going to break every cross-link between them. Running this before and after turned
that from a hope into a check.

IT ALSO FOUND A BUG THAT PREDATED THE MOVE. Doc links inside the ops headers read
`../../../ARCHITECTURE.md`, which from their depth resolved to a file that never
existed. They had been broken since they were written and nobody had checked, which is
the whole argument for having this script rather than being careful.

IT EARNED ITSELF AGAIN when the library moved to the repository root (#35). Every
header's depth changed by one level at once, and eight links across six files broke
silently -- the kind of breakage a move makes invisible precisely because it is
uniform. This script named all eight, which is why a layout change runs it.

Exits non-zero if anything is broken, so it can gate a commit.

Usage: python3 scripts/check_links.py
"""

import os, re, subprocess, sys, collections

files = [f for f in subprocess.check_output(['git','ls-files']).decode().split('\n')
         if f and re.search(r'\.(md|hpp|cpp|sh|py)$', f) and not f.startswith('docs/api/')]
bad = collections.Counter(); detail = collections.defaultdict(list)
for f in files:
    try: s = open(f, encoding='utf-8').read()
    except Exception: continue
    d = os.path.dirname(f) or '.'
    for link in re.findall(r'\]\(([^)\s]+)\)', s):
        if link.startswith(('http://','https://','#','mailto:')): continue
        path = link.split('#')[0]
        if not path: continue
        # C++ lambdas read as ](int) etc.; a link worth checking names a file
        if not re.search(r'\.[A-Za-z0-9]{1,5}$', path): continue
        if not os.path.exists(os.path.normpath(os.path.join(d, path))):
            bad[f] += 1; detail[f].append(path)
total = sum(bad.values())
print("broken links:", total, "in", len(bad), "files")
for f, n in bad.most_common(20):
    print(f"  {n:4d}  {f}")
    for p in sorted(set(detail[f]))[:5]: print(f"          -> {p}")
sys.exit(1 if total else 0)
