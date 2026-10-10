#!/usr/bin/env python3
"""m7_dump_compare.py -- a board's float outputs against the host's, value by value.

The integer outputs of the board images are held to the host's by digest: equal or not.
A float output can differ in its last bits for a reason that is not a defect -- the
Cortex-M7 build contracts multiply-adds into fused ones (GCC's default
`-ffp-contract=fast`) where the x86-64 baseline has no FMA to contract into -- so for
those a digest says only THAT they differ. This says by how much.

    scripts/m7_dump_compare.py BOARD HOST

BOARD is a log holding the board's dump lines (a `launches` log of ESSENTIALDUMP or
SUBPIXDUMP, or `m7_host.py run` output); HOST is what `m7_reference essential --dump` or
`m7_reference subpix --dump` printed. The lines:

    E <count> <nine doubles as 16 hex digits>     the essential matrix, per scene
    M <count> <one 0/1 per correspondence>        its inlier mask
    S <x,y float bits, 16 hex digits each ...>    cornerSubPix's seeds
    P <x,y float bits, 16 hex digits each ...>    the refined points

A log with several launches repeats its dump; every repeat is held to the host's.

EXIT CODES
    0   compared (whatever the differences -- they are reported, not judged)
    1   a line the host printed is missing from the board's, or vice versa
    2   bad invocation
"""

import struct
import sys


def load(path):
    """Every dump line, as a list per key: (kind, count) -> [fields, ...]."""
    out = {}
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if len(t) < 2 or t[0] not in ('E', 'M', 'S', 'P'):
                continue
            key = (t[0], int(t[1])) if t[0] in ('E', 'M') else (t[0], 0)
            fields = t[2:] if t[0] in ('E', 'M') else t[1:]
            out.setdefault(key, []).append(fields)
    return out


def double(h):
    return struct.unpack('>d', bytes.fromhex(h))[0]


def point(h):
    return (struct.unpack('>f', bytes.fromhex(h[:8]))[0],
            struct.unpack('>f', bytes.fromhex(h[8:]))[0])


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    board, host = load(sys.argv[1]), load(sys.argv[2])
    missing = sorted(set(host) ^ set(board))
    for key in missing:
        print(f'{key[0]} {key[1]}: only in the {"host" if key in host else "board"} dump')
    for key in sorted(set(host) & set(board)):
        want = host[key][0]
        for n, got in enumerate(board[key], 1):
            tag = f'{key[0]} {key[1]:>5}' if key[0] in ('E', 'M') else f'{key[0]}      '
            tag += f' (launch {n})' if len(board[key]) > 1 else ''
            if key[0] == 'E':
                same = sum(1 for a, b in zip(got, want) if a == b)
                diff = max(abs(double(a) - double(b)) for a, b in zip(got, want))
                print(f'{tag}: {same} of 9 entries bit-equal, largest difference {diff:.3e}')
            elif key[0] == 'M':
                a, b = got[0], want[0]
                differ = sum(1 for x, y in zip(a, b) if x != y)
                print(f'{tag}: masks {"equal" if a == b else f"differ on {differ}"}, '
                      f'{a.count("1")} inliers on the board, {b.count("1")} on the host')
            else:
                pa, pb = [point(v) for v in got], [point(v) for v in want]
                d = [((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                     for (x0, y0), (x1, y1) in zip(pa, pb)]
                same = sum(1 for a, b in zip(got, want) if a == b)
                print(f'{tag}: {same} of {len(want)} points bit-equal, '
                      f'largest difference {max(d) if d else 0.0:.3e} px')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
