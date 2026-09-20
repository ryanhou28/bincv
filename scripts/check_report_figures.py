#!/usr/bin/env python3
"""check_report_figures.py -- every figure in a summary table is in the report it cites.

WHY THIS EXISTS. docs/reports/README.md advertised the CUDA backend at "2.1x faster than
cv::cuda::StereoBM at 5x less device memory" for two rounds after the measurements read
10.8x and 6.857x. Nothing connected the summary to its source, so the front door
contradicted the report and no gate noticed. The same drift is possible on every summary
row in the repository: the root README, the reports index, and backends/cuda/README.md
all quote figures that are measured somewhere else.

WHAT IT CHECKS. A summary table opts in with a directive comment on the line above it:

    <!-- figure-check values="OpenCV|binCV" source="source" -->

`values` names the header cells whose numeric tokens are figures. `source` names either
the header cell holding a markdown link to the report (one link, a .md file, resolved
relative to the citing file) or, with a leading @, a fixed repository path for the whole
table. Every numeric token in a value cell of every body row must occur in that report,
standalone -- not as a fragment of a longer number. A token written with a trailing
multiplication sign or per cent sign must occur in the report with the same sign, so
"6.857x" is not satisfied by a bare 6.857 somewhere in the file.

Naming the columns is what keeps this from being a fragile regex. The script never
guesses which cells are figures and which are conditions: "640x480", "kNN=2" and "3x3"
live in columns `values` does not name, so they are never looked up.

WHAT IT CANNOT CATCH. Presence is file-scoped. A figure moved to a different row, a
different table or a different operation inside the same report still passes, because
the script asks whether the report contains the number and not whether it contains it
in the row the summary claims. Catching that needs per-row anchors in the reports, which
they do not have. It also says nothing about whether a figure is correct -- only that the
summary and the report have not drifted apart. And it checks the tables that opted in:
a new summary table with no directive is not checked, which is why the directive sits
with the table rather than in a list somewhere else.

Exits non-zero if anything is unsourced, so it can gate a commit.

Usage: python3 scripts/check_report_figures.py
"""

import os, re, subprocess, sys

DIRECTIVE = re.compile(r'<!--\s*figure-check\s+(.*?)-->\s*$')
ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
# A figure: digits, optionally grouped with commas, optionally a decimal tail.
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
MDLINK = re.compile(r'\]\(([^)\s]+)\)')


def split_cells(line):
    """Split a markdown table row, respecting `code spans` that may contain a pipe."""
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    cells, cur, i = [], '', 0
    while i < len(s):
        c = s[i]
        if c == '`':
            j = s.find('`', i + 1)
            j = len(s) - 1 if j < 0 else j
            cur += s[i:j + 1]
            i = j + 1
        elif c == '|':
            cells.append(cur.strip())
            cur = ''
            i += 1
        else:
            cur += c
            i += 1
    cells.append(cur.strip())
    return cells


def header_key(cell):
    """The header cell as a lookup key: no emphasis, no code ticks, no case."""
    return re.sub(r'[*`]', '', cell).strip().lower()


def figures(cell):
    """Numeric tokens in a cell, each with the sign that follows it if it is x or %."""
    out = []
    for m in NUMBER.finditer(cell):
        tok = m.group(0).rstrip(',')
        if not tok:
            continue
        rest = cell[m.start() + len(tok):]
        # A trailing multiplication or per cent sign marks a ratio. "5×5" is a
        # geometry rather than a ratio, so the sign counts only when no digit follows it.
        sign = rest[:1] if rest[:1] in ('×', '%') and not rest[1:2].isdigit() else ''
        out.append((tok.replace(',', ''), sign))
    return out


def present(token, sign, haystack):
    """The report contains this figure, standalone rather than inside a longer number."""
    return re.search(r'(?<![\d.])' + re.escape(token) + re.escape(sign) + r'(?![\d])',
                     haystack) is not None


def load(path, cache={}):
    if path not in cache:
        try:
            # Commas group digits in the reports too; compare without them.
            cache[path] = open(path, encoding='utf-8').read().replace(',', '')
        except OSError:
            cache[path] = None
    return cache[path]


def tables(path):
    """Yield (line number, attrs, header cells, body rows) for each opted-in table."""
    lines = open(path, encoding='utf-8').read().split('\n')
    for i, line in enumerate(lines):
        m = DIRECTIVE.search(line)
        if not m:
            continue
        attrs = dict(ATTR.findall(m.group(1)))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j + 1 >= len(lines) or not lines[j].strip().startswith('|'):
            yield i + 1, attrs, None, None
            continue
        header = split_cells(lines[j])
        body, k = [], j + 2          # j + 1 is the delimiter row
        while k < len(lines) and lines[k].strip().startswith('|'):
            body.append((k + 1, split_cells(lines[k])))
            k += 1
        yield i + 1, attrs, header, body


def main():
    files = [f for f in subprocess.check_output(['git', 'ls-files']).decode().split('\n')
             if f.endswith('.md') and not f.startswith('docs/api/')]
    problems, checked_tables, checked_figures = [], 0, 0

    for path in files:
        for lineno, attrs, header, body in tables(path):
            where = f'{path}:{lineno}'
            if header is None:
                problems.append(f'{where}: figure-check directive has no table under it')
                continue
            if 'values' not in attrs or 'source' not in attrs:
                problems.append(f'{where}: directive needs both values="..." and source="..."')
                continue
            keys = [header_key(c) for c in header]
            wanted = [v.strip().lower() for v in attrs['values'].split('|') if v.strip()]
            missing = [w for w in wanted if w not in keys]
            if missing:
                problems.append(f'{where}: no column named {missing} in {keys}')
                continue
            value_at = [keys.index(w) for w in wanted]

            fixed_source = attrs['source'].startswith('@')
            if fixed_source:
                src_col = None
                fixed_path = attrs['source'][1:]
            elif attrs['source'].strip().lower() in keys:
                src_col = keys.index(attrs['source'].strip().lower())
            else:
                problems.append(f'{where}: no source column named "{attrs["source"]}"')
                continue

            checked_tables += 1
            for rowno, cells in body:
                if len(cells) != len(header):
                    problems.append(f'{path}:{rowno}: row has {len(cells)} cells, '
                                    f'header has {len(header)}')
                    continue
                if fixed_source:
                    report = fixed_path
                else:
                    links = MDLINK.findall(cells[src_col])
                    md = [l.split('#')[0] for l in links if l.split('#')[0].endswith('.md')]
                    if len(md) != 1:
                        problems.append(f'{path}:{rowno}: source cell must hold exactly one '
                                        f'.md link, found {len(md)}')
                        continue
                    report = os.path.normpath(os.path.join(os.path.dirname(path) or '.', md[0]))
                text = load(report)
                if text is None:
                    problems.append(f'{path}:{rowno}: cited report {report} does not exist')
                    continue
                for col in value_at:
                    for token, sign in figures(cells[col]):
                        checked_figures += 1
                        if not present(token, sign, text):
                            problems.append(
                                f'{path}:{rowno}: {token}{sign} ({header_key(header[col])}) '
                                f'is not in {report}')

    print(f'summary tables checked: {checked_tables}')
    print(f'figures checked: {checked_figures}')
    print(f'unsourced figures: {len(problems)}')
    for p in problems[:40]:
        print(f'  {p}')
    if len(problems) > 40:
        print(f'  ... and {len(problems) - 40} more')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
