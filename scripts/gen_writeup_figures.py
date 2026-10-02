#!/usr/bin/env python3
"""gen_writeup_figures.py -- draws every figure in docs/writeups/figures/.

WHY A SCRIPT. The diagrams in the writeups show bit patterns, and a hand-drawn bit
pattern is a claim nobody checks: a carry drawn into the wrong word, or an adder whose
output does not add, reads as correct. Here every pattern a figure shows is COMPUTED --
the shifted row is the source row shifted, the sum planes are the sum -- so a figure can
be wrong only if the arithmetic in this file is, and that arithmetic is a few lines per
figure.

WHY NO PLOTTING LIBRARY. The output has to be byte-identical from one run to the next so
`--check` can gate it, and plotting libraries stamp dates and random ids into SVG. The
standard library writes the same bytes every time, and CI needs nothing installed.

DATA CHARTS READ THE REPORTS, NEVER A LITERAL. A chart of measured figures parses the
report table those figures are published in. The report is held to its logs by
check_figure_staleness.py; `--check` then holds the chart to the report, so a re-taken
figure that is not re-drawn fails a gate instead of leaving a picture of an old number.

Usage:
    python3 scripts/gen_writeup_figures.py           # write every figure
    python3 scripts/gen_writeup_figures.py --check   # exit 1 if any figure is out of date
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'writeups', 'figures')

# One palette for every figure, redefined for a dark reader. The figure carries its own
# background so its text never lands on a page colour it was not drawn for.
STYLE = """
.bg{fill:#ffffff}
.fg{fill:#1f2328}
.muted{fill:#59636e}
.on{fill:#2f6fdb}
.off{fill:#eef1f5}
.hi{fill:#d4620f}
.hisoft{fill:#fbe3cf}
.neg{fill:#d4620f}
.pad{fill:#d7dbe0}
.cell{stroke:#aeb6bf;stroke-width:0.75}
.word{fill:none;stroke:#1f2328;stroke-width:2}
.box{fill:none;stroke:#59636e;stroke-width:1;stroke-dasharray:3 3}
.ring{fill:none;stroke:#d4620f;stroke-width:2.5}
.edge{stroke:#59636e;stroke-width:1.25;fill:none}
.arrowhead{fill:#59636e}
.hiedge{stroke:#d4620f;stroke-width:2;fill:none}
.hiarrowhead{fill:#d4620f}
.bar1{fill:#aeb6bf}
.bar2{fill:#2f6fdb}
.s1{stroke:#2f6fdb;stroke-width:2.5;fill:none}
.s2{stroke:#d4620f;stroke-width:2.5;fill:none}
.d1{fill:#2f6fdb}
.d2{fill:#d4620f}
.gridline{stroke:#d7dbe0;stroke-width:1}
.refline{stroke:#1f2328;stroke-width:1.5;stroke-dasharray:5 4}
.axis{stroke:#59636e;stroke-width:1}
.halo{paint-order:stroke;stroke:#ffffff;stroke-width:4px;stroke-linejoin:round}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.oninv{fill:#ffffff}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117}
.fg{fill:#e6edf3}
.muted{fill:#9198a1}
.on{fill:#4493f8}
.off{fill:#1c2129}
.hi{fill:#f0883e}
.hisoft{fill:#4a2a12}
.neg{fill:#f0883e}
.pad{fill:#3d444d}
.cell{stroke:#3d444d}
.word{stroke:#e6edf3}
.box{stroke:#9198a1}
.ring{stroke:#f0883e}
.edge{stroke:#9198a1}
.arrowhead{fill:#9198a1}
.hiedge{stroke:#f0883e}
.hiarrowhead{fill:#f0883e}
.bar1{fill:#59636e}
.bar2{fill:#4493f8}
.s1{stroke:#4493f8}
.s2{stroke:#f0883e}
.d1{fill:#4493f8}
.d2{fill:#f0883e}
.gridline{stroke:#30363d}
.refline{stroke:#e6edf3}
.axis{stroke:#9198a1}
.halo{stroke:#0d1117}
.oninv{fill:#0d1117}
}
"""


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def num(v):
    """Coordinates print as integers where they are, so the output is stable and short."""
    return str(int(v)) if float(v).is_integer() else ('%.2f' % v).rstrip('0').rstrip('.')


class Svg:
    def __init__(self, width, height, label):
        self.w, self.h, self.label = width, height, label
        self.parts = []

    def rect(self, x, y, w, h, cls, rx=0):
        r = ' rx="%s"' % num(rx) if rx else ''
        self.parts.append('<rect x="%s" y="%s" width="%s" height="%s" class="%s"%s/>'
                          % (num(x), num(y), num(w), num(h), cls, r))

    def text(self, x, y, s, cls='fg', size=13, anchor='start', weight=None, mono=False):
        c = cls + (' mono' if mono else '')
        wt = ' font-weight="%s"' % weight if weight else ''
        sp = ' xml:space="preserve"' if '  ' in s else ''
        self.parts.append('<text x="%s" y="%s" font-size="%s" text-anchor="%s" class="%s"%s%s>%s</text>'
                          % (num(x), num(y), num(size), anchor, c, wt, sp, esc(s)))

    def line(self, x1, y1, x2, y2, cls='edge'):
        self.parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" class="%s"/>'
                          % (num(x1), num(y1), num(x2), num(y2), cls))

    def arrow(self, x1, y1, x2, y2, hi=False):
        """A straight arrow whose head ends exactly at (x2, y2)."""
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / n, dy / n
        bx, by = x2 - 7 * ux, y2 - 7 * uy
        self.line(x1, y1, bx, by, 'hiedge' if hi else 'edge')
        px, py = -uy * 4, ux * 4
        self.parts.append('<path d="M%s %sL%s %sL%s %sZ" class="%s"/>'
                          % (num(x2), num(y2), num(bx + px), num(by + py), num(bx - px),
                             num(by - py), 'hiarrowhead' if hi else 'arrowhead'))

    def render(self):
        head = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d" role="img" aria-label="%s">\n<style>%s</style>\n'
                % (self.w, self.h, self.w, self.h, esc(self.label), STYLE))
        body = '<rect width="100%%" height="100%%" class="bg" rx="8"/>\n' + '\n'.join(self.parts)
        return head + body + '\n</svg>\n'


def bitrow(svg, x, y, bits, cell, word=None, cls_of=None, labels=None, label_cls='oninv'):
    """One row of pixel cells. `word` draws a heavy border every `word` cells;
    `cls_of(i, b)` overrides a cell's fill; `labels` writes text into each cell."""
    for i, b in enumerate(bits):
        cls = cls_of(i, b) if cls_of else None
        cls = cls or ('on' if b else 'off')
        svg.rect(x + i * cell, y, cell, cell, cls + ' cell')
        if labels is not None:
            lab = labels[i]
            tc = label_cls if cls in ('on', 'neg', 'hi') else 'fg'
            svg.text(x + i * cell + cell / 2, y + cell / 2 + 4.5, str(lab), tc, 12, 'middle', mono=True)
    if word:
        for s in range(0, len(bits), word):
            n = min(word, len(bits) - s)
            svg.rect(x + s * cell, y, n * cell, cell, 'word')


def grid(svg, x, y, rows, cell, cls_of=None):
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cls = cls_of(r, c, v) if cls_of else ('on' if v else 'off')
            svg.rect(x + c * cell, y + r * cell, cell, cell, cls + ' cell')


# ---------------------------------------------------------------------------------------
# Premise figures


def fig_bytes_vs_bits():
    mask = ['0001111000000000',
            '0011111100011000',
            '0011111100111100',
            '0001111000011000']
    rows = [[int(c) for c in r] for r in mask]
    W, cell = 8, 20
    svg = Svg(860, 404, 'A 16 by 4 binary mask stored as 64 bytes and as eight 8-bit words, with eight pixels of '
                        'one row expanded both ways: eight bytes of which one bit each carries the image, and one '
                        '8-bit word')

    ax, bx, gy = 30, 470, 62
    svg.text(ax, 30, 'One byte per pixel (CV_8U)', 'fg', 14, weight='600')
    svg.text(ax, 48, '64 pixels = 64 bytes', 'muted', 12)
    grid(svg, ax, gy, rows, cell)
    for r in range(4):
        for c in range(16):
            svg.rect(ax + c * cell, gy + r * cell, cell, cell, 'word')

    svg.text(bx, 30, 'One bit per pixel, 8-bit words', 'fg', 14, weight='600')
    svg.text(bx, 48, '64 pixels = 8 words = 8 bytes', 'muted', 12)
    grid(svg, bx, gy, rows, cell)
    for r in range(4):
        for s in range(0, 16, W):
            svg.rect(bx + s * cell, gy + r * cell, W * cell, cell, 'word')
    for s in range(0, 16, W):
        svg.text(bx + (s + W / 2) * cell, gy + 4 * cell + 16, 'word %d' % (s // W), 'muted', 11, 'middle')

    r = 1
    bits = rows[r][:W]
    src_y = gy + r * cell
    zy = 236

    # Left zoom: the same eight pixels as eight bytes, every bit of every byte drawn.
    bc = 6
    svg.rect(ax, src_y, W * cell, cell, 'ring')
    svg.arrow(ax + W * cell / 2, gy + 4 * cell + 8, ax + W * cell / 2 + 40, zy - 30, hi=True)
    svg.text(ax, zy - 12, 'Row 1, pixels 0\u20137: eight bytes', 'fg', 13, weight='600')
    for i, b in enumerate(bits):
        x = ax + i * W * bc
        for k in range(W):
            lsb = k == W - 1
            cls = ('on' if b else 'off') if lsb else 'pad'
            svg.rect(x + k * bc, zy, bc, 30, cls + ' cell')
        svg.rect(x, zy, W * bc, 30, 'word')
        svg.text(x + W * bc / 2, zy + 46, '0x%02X' % b, 'fg', 11, 'middle', mono=True)
        svg.text(x + W * bc / 2, zy + 60, 'px %d' % i, 'muted', 10, 'middle', mono=True)
    svg.text(ax, zy + 90, '64 bits stored. One bit in each byte carries the pixel;', 'fg', 12)
    svg.text(ax, zy + 106, 'the other seven, drawn grey, are always zero.', 'fg', 12)

    # Right zoom: the same eight pixels as one word.
    zc = 30
    svg.rect(bx, src_y, W * cell, cell, 'ring')
    svg.arrow(bx + W * cell / 2, gy + 4 * cell + 24, bx + W * zc / 2, zy - 30, hi=True)
    svg.text(bx, zy - 12, 'Row 1, pixels 0\u20137: one 8-bit word', 'fg', 13, weight='600')
    for i, b in enumerate(bits):
        svg.rect(bx + i * zc, zy, zc, 30, ('on' if b else 'off') + ' cell')
        svg.text(bx + i * zc + zc / 2, zy + 20, str(b), 'oninv' if b else 'fg', 13, 'middle', mono=True)
        svg.text(bx + i * zc + zc / 2, zy + 46, str(i), 'muted', 10, 'middle', mono=True)
    svg.rect(bx, zy, W * zc, 30, 'word')
    svg.text(bx - 8, zy + 46, 'bit', 'muted', 10, 'end')
    value = sum(b << i for i, b in enumerate(bits))
    msb = ''.join(str(b) for b in reversed(bits))
    svg.text(bx, zy + 90, '8 bits stored. Pixel x is bit x of the row.', 'fg', 12)
    svg.text(bx, zy + 106, 'As a number: 0b%s = 0x%02X' % (msb, value), 'fg', 12, mono=True)
    svg.text(bx, zy + 122, 'Pixel 0 is the lowest bit, so the number reads right to left.', 'muted', 11)

    svg.text(30, 388, 'Drawn with 8-bit words for legibility; the default word is uint32_t, 32 pixels.', 'muted', 11)
    return svg


def fig_and():
    a = [int(c) for c in '0011110000111100']
    b = [int(c) for c in '0000111111110000']
    d = [x & y for x, y in zip(a, b)]
    W, cell, n = 8, 19, 16
    svg = Svg(860, 300, 'ANDing two 16-pixel rows: sixteen instructions when each pixel is a byte, two when each '
                        'pixel is a bit in an 8-bit word')
    lx, ax, bx, y0, dy = 64, 80, 480, 58, 34
    svg.text(ax, 26, 'One byte per pixel', 'fg', 14, weight='600')
    svg.text(ax, 43, 'each AND handles one pixel', 'muted', 11)
    svg.text(bx, 26, 'One bit per pixel, 8-bit words', 'fg', 14, weight='600')
    svg.text(bx, 43, 'each AND handles a whole word: 8 pixels here, 32 in a uint32_t', 'muted', 11)
    for k, (name, row) in enumerate((('a', a), ('b', b), ('a & b', d))):
        y = y0 + k * dy + (8 if k == 2 else 0)
        svg.text(lx, y + 14, name, 'fg', 13, 'end', mono=True)
        bitrow(svg, ax, y, row, cell)
        for i in range(n):
            svg.rect(ax + i * cell, y, cell, cell, 'word')
        bitrow(svg, bx, y, row, cell, word=W)
    yl = y0 + 2 * dy + 2
    svg.line(ax, yl, ax + n * cell, yl, 'edge')
    svg.line(bx, yl, bx + n * cell, yl, 'edge')
    yc = y0 + 3 * dy + 28
    svg.text(ax, yc, '16 AND instructions', 'hi', 14, weight='600')
    svg.text(bx, yc, '2 AND instructions', 'hi', 14, weight='600')
    yc += 34
    svg.text(30, yc, 'bytes:', 'muted', 12)
    svg.text(90, yc, 'for (x = 0; x < 640; ++x) d[x] = a[x] & b[x];   // 640 ANDs per VGA row', 'fg', 12, mono=True)
    svg.text(30, yc + 20, 'bits:', 'muted', 12)
    svg.text(90, yc + 20, 'for (i = 0; i <  20; ++i) d[i] = a[i] & b[i];   //  20 ANDs, uint32_t words', 'fg', 12,
             mono=True)
    svg.text(30, yc + 46, 'A compiler vectorizes the byte loop too, 16 or 32 bytes per instruction; the same vector '
             'registers hold eight times as many pixels as bits.', 'muted', 11)
    return svg


def row_words(bits, W):
    return [sum(b << i for i, b in enumerate(bits[s:s + W])) for s in range(0, len(bits), W)]


def words_row(words, W, width):
    bits = []
    for w in words:
        bits += [(w >> i) & 1 for i in range(W)]
    return bits[:width]


def fig_shift_or():
    W, width, cell = 8, 16, 22
    src = [0] * width
    for x in (W - 1, W, 13):
        src[x] = 1
    w = row_words(src, W)
    mask = (1 << W) - 1
    right = [((w[i] << 1) | (w[i - 1] >> (W - 1) if i > 0 else 0)) & mask for i in range(len(w))]
    left = [((w[i] >> 1) | ((w[i + 1] << (W - 1)) & mask if i + 1 < len(w) else 0)) for i in range(len(w))]
    out = [w[i] | left[i] | right[i] for i in range(len(w))]
    r_bits, l_bits, o_bits = (words_row(v, W, width) for v in (right, left, out))

    # The carried pixels, computed rather than named: set in the shifted row, and
    # their source pixel sits in the other word.
    carried_r = [x for x in range(width) if r_bits[x] and x % W == 0]
    carried_l = [x for x in range(width) if l_bits[x] and x % W == W - 1]

    svg = Svg(900, 262, 'Horizontal dilation of a packed row as two shifts and an OR, '
                        'with one pixel carried across a word boundary')
    gx, y0, dy = 210, 40, 52
    svg.text(gx, 22, 'word 0', 'muted', 11)
    svg.text(gx + W * cell, 22, 'word 1', 'muted', 11)
    labels = [('source', ''),
              ('shifted right one pixel', ''),
              ('shifted left one pixel', ''),
              ('dilated, 1×3', '')]
    formulas = ['w[i]',
                '(w[i] << 1) | (w[i−1] >> (W−1))',
                '(w[i] >> 1) | (w[i+1] << (W−1))',
                'w[i] | left | right']
    rowsbits = [src, r_bits, l_bits, o_bits]
    for k, bits in enumerate(rowsbits):
        y = y0 + k * dy
        svg.text(gx - 14, y + cell / 2 + 5, labels[k][0], 'fg', 13, 'end')

        def cls_of(i, b, k=k):
            if k == 1 and i in carried_r:
                return 'hi'
            if k == 2 and i in carried_l:
                return 'hi'
            return None
        bitrow(svg, gx, y, bits, cell, word=W, cls_of=cls_of)
        svg.text(gx + width * cell + 22, y + cell / 2 + 5, formulas[k], 'fg', 13, mono=True)
    for x in [x - 1 for x in carried_r] + [x + 1 for x in carried_l]:
        svg.rect(gx + x * cell, y0, cell, cell, 'ring')
    svg.text(gx, y0 + 4 * dy - 4, 'Orange: a pixel crossing the word boundary, ringed where it starts. The second '
             'term of each formula carries it in.', 'muted', 11)
    svg.text(gx, y0 + 4 * dy + 12, 'Every line is one operation per word, W pixels at a time.', 'muted', 11)
    return svg


def fig_bitplanes():
    vals = [5, 3, 0, 7, 2, 6, 1, 4]
    N, cell = 3, 34
    svg = Svg(820, 350, 'Eight 3-bit pixels stored as packed 3-bit fields and as three bit-planes')
    gx = 190
    y = 34
    svg.text(gx - 14, y + cell / 2 + 5, 'pixel values', 'fg', 13, 'end')
    for i, v in enumerate(vals):
        cls = 'hisoft' if i == 0 else 'off'
        svg.rect(gx + i * cell, y, cell, cell, cls + ' cell')
        svg.text(gx + i * cell + cell / 2, y + cell / 2 + 5, str(v), 'fg', 15, 'middle', mono=True)
        svg.text(gx + i * cell + cell / 2, y - 6, format(v, '03b'), 'muted', 10, 'middle', mono=True)

    svg.text(gx, y + cell + 30, 'Bit-planes: plane i holds bit i of every pixel', 'fg', 14, weight='600')
    py = y + cell + 44
    for p in range(N):
        bits = [(v >> p) & 1 for v in vals]
        yy = py + p * (cell + 8)
        svg.text(gx - 14, yy + cell / 2 + 5, 'plane %d (weight %d)' % (p, 1 << p), 'fg', 13, 'end')
        bitrow(svg, gx, yy, bits, cell, word=8, labels=bits)
        svg.rect(gx, yy, cell, cell, 'ring')
    svg.text(gx + 8 * cell + 20, py + cell / 2 + 5, 'three 8-bit words', 'muted', 12)
    svg.text(gx + 8 * cell + 20, py + cell / 2 + 23, 'every pixel sits at the', 'muted', 12)
    svg.text(gx + 8 * cell + 20, py + cell / 2 + 41, 'same bit in every plane', 'muted', 12)
    svg.text(gx + 8 * cell + 20, py + 2 * (cell + 8) + cell / 2 + 5, 'pixel 0: 5 = 101', 'hi', 12, mono=True)

    fy = py + N * (cell + 8) + 36
    fc = 22
    svg.text(gx, fy - 12, 'Packed fields: three bits per pixel, side by side', 'fg', 14, weight='600')
    bits = []
    owner = []
    for i, v in enumerate(vals):
        for b in range(N):
            bits.append((v >> b) & 1)
            owner.append(i)
    for j, b in enumerate(bits):
        svg.rect(gx + j * fc, fy, fc, fc, ('on' if b else 'off') + ' cell')
    for i in range(1, len(vals)):
        svg.line(gx + i * N * fc, fy - 5, gx + i * N * fc, fy + fc + 5, 'hiedge')
    for s in range(0, len(bits), 8):
        svg.rect(gx + s * fc, fy, 8 * fc, fc, 'word')
    for i in range(len(vals)):
        svg.text(gx + (i * N + N / 2) * fc, fy + fc + 14, str(i), 'muted', 10, 'middle', mono=True)
    svg.text(gx - 14, fy + fc / 2 + 5, '3 bytes', 'fg', 13, 'end')
    svg.text(gx, fy + fc + 32, 'Orange lines divide the fields. Fields 2 and 5 straddle a word boundary, and adding '
             'two images', 'muted', 11)
    svg.text(gx, fy + fc + 48, 'this way needs a mask and a carry guard for every field.', 'muted', 11)
    return svg


def fig_adder():
    A = [3, 1, 2, 0, 3, 2, 1, 3]
    B = [2, 3, 1, 0, 3, 0, 1, 1]
    n = len(A)
    W = 8

    def plane(vals, p):
        return sum(((v >> p) & 1) << i for i, v in enumerate(vals))

    a0, a1, b0, b1 = plane(A, 0), plane(A, 1), plane(B, 0), plane(B, 1)
    s0 = a0 ^ b0
    c0 = a0 & b0
    t = a1 ^ b1
    s1 = t ^ c0
    c1 = (a1 & b1) | (c0 & t)
    s2 = c1
    S = [((s0 >> i) & 1) | (((s1 >> i) & 1) << 1) | (((s2 >> i) & 1) << 2) for i in range(n)]
    assert S == [a + b for a, b in zip(A, B)], 'the adder in this figure does not add'

    rows = [('A, plane 0', 'a0', a0, None, ''),
            ('A, plane 1', 'a1', a1, None, ''),
            ('B, plane 0', 'b0', b0, None, ''),
            ('B, plane 1', 'b1', b1, None, ''),
            ('sum, plane 0', 's0', s0, 'a0 ^ b0', '1 op'),
            ('carry', 'c0', c0, 'a0 & b0', '1 op'),
            ('sum, plane 1', 's1', s1, 'a1 ^ b1 ^ c0', '2 ops'),
            ('sum, plane 2', 's2', s2, '(a1 & b1) | (c0 & (a1 ^ b1))', '3 ops, a1 ^ b1 reused')]
    cell, gx, gy, dy = 26, 180, 60, 32
    svg = Svg(900, 90 + len(rows) * dy + 76, 'A bit-sliced adder adding two 2-bit images, eight pixels at once, '
                                              'in seven word operations')
    svg.text(gx, 22, 'Adding two 2-bit images, all eight pixels at once', 'fg', 14, weight='600')
    for i in range(n):
        svg.text(gx + i * cell + cell / 2, gy - 22, 'A=%d' % A[i], 'muted', 10, 'middle', mono=True)
        svg.text(gx + i * cell + cell / 2, gy - 9, 'B=%d' % B[i], 'muted', 10, 'middle', mono=True)
    for k, (lab, name, word, expr, cost) in enumerate(rows):
        y = gy + k * dy + (14 if k >= 4 else 0)
        bits = [(word >> i) & 1 for i in range(n)]
        svg.text(gx - 14, y + cell / 2 + 5, lab, 'fg', 13, 'end')
        bitrow(svg, gx, y, bits, cell, word=W, labels=bits)
        if expr:
            svg.text(gx + n * cell + 20, y + cell / 2 + 5, '%s = %s' % (name, expr), 'fg', 13, mono=True)
            svg.text(880, y + cell / 2 + 5, cost, 'muted', 11, 'end')
    svg.line(gx - 150, gy + 4 * dy + 5, 880, gy + 4 * dy + 5, 'edge')
    col = 0
    svg.rect(gx + col * cell, gy, cell, 4 * dy - (dy - cell), 'ring')
    svg.rect(gx + col * cell, gy + 4 * dy + 14, cell, 4 * dy - (dy - cell), 'ring')
    y = gy + len(rows) * dy + 14 + 18
    svg.text(gx - 14, y + 12, 'sum', 'fg', 13, 'end')
    for i in range(n):
        svg.rect(gx + i * cell, y, cell, cell, ('hisoft' if i == col else 'off') + ' cell')
        svg.text(gx + i * cell + cell / 2, y + cell / 2 + 5, str(S[i]), 'fg', 13, 'middle', mono=True)
    svg.text(gx + n * cell + 20, y + cell / 2 + 5,
             'pixel 0: %d + %d = %d = %s, read down the ringed column' % (A[col], B[col], S[col], format(S[col], '03b')),
             'hi', 12, mono=True)
    svg.text(gx - 150, y + cell + 30, 'Seven word operations add every pixel in the word: 8 here, 32 in a uint32_t, '
             'with the same seven instructions. No carry ever crosses between pixels.', 'muted', 11)
    return svg


def ternary_patch():
    """A diagonal bar, its [-1, 0, 1] derivatives, and nothing near the border, so the
    border rule cannot change a single value the figure shows."""
    H = Wd = 12

    def inside(y, x):
        u = ((x - 5.5) + (y - 5.5)) / 2 ** 0.5
        v = ((x - 5.5) - (y - 5.5)) / 2 ** 0.5
        return (u / 4.5) ** 2 + (v / 2.6) ** 2 <= 1
    img = [[1 if inside(y, x) else 0 for x in range(Wd)] for y in range(H)]
    assert not any(img[y][x] for y in range(H) for x in (0, Wd - 1)) and not any(img[0] + img[-1])

    def at(y, x):
        return img[y][x] if 0 <= y < H and 0 <= x < Wd else 0

    # filter2D correlates: the derivative is the right neighbour minus the left one.
    ix = [[at(y, x + 1) - at(y, x - 1) for x in range(Wd)] for y in range(H)]
    iy = [[at(y + 1, x) - at(y - 1, x) for x in range(Wd)] for y in range(H)]
    return img, ix, iy


def fig_ternary():
    img, ix, iy = ternary_patch()
    H, Wd = len(img), len(img[0])
    prod = [[ix[y][x] * iy[y][x] for x in range(Wd)] for y in range(H)]

    # The popcount spelling, evaluated on the planes rather than on the integers above.
    magx = [[1 if v else 0 for v in r] for r in ix]
    magy = [[1 if v else 0 for v in r] for r in iy]
    sgnx = [[1 if v < 0 else 0 for v in r] for r in ix]
    sgny = [[1 if v < 0 else 0 for v in r] for r in iy]
    agree = sum(magx[y][x] & magy[y][x] & (1 - (sgnx[y][x] ^ sgny[y][x])) for y in range(H) for x in range(Wd))
    oppose = sum(magx[y][x] & magy[y][x] & (sgnx[y][x] ^ sgny[y][x]) for y in range(H) for x in range(Wd))
    sxx = sum(map(sum, magx))
    syy = sum(map(sum, magy))
    assert agree - oppose == sum(map(sum, prod))
    assert sxx == sum(v * v for r in ix for v in r)

    cell, gap, top = 14, 34, 56
    pw = Wd * cell
    svg = Svg(4 * pw + 3 * gap + 60, top + H * cell + 150,
              'A binary diagonal bar, its ternary x and y derivatives, and their product, '
              'with the covariance sums as population counts')

    def tern(r, c, v):
        return 'on' if v > 0 else ('neg' if v < 0 else 'off')
    titles = [('image', 'binary'), ('Ix', 'right − left'), ('Iy', 'below − above'), ('Ix·Iy', 'product')]
    data = [img, ix, iy, prod]
    for k in range(4):
        x = 30 + k * (pw + gap)
        svg.text(x, 26, titles[k][0], 'fg', 14, weight='600')
        svg.text(x, 43, titles[k][1], 'muted', 11)
        grid(svg, x, top, data[k], cell, None if k == 0 else tern)
    ly = top + H * cell + 22
    x = 30 + pw + gap
    svg.rect(x, ly - 10, 12, 12, 'on cell')
    svg.text(x + 18, ly, '+1', 'fg', 12)
    svg.rect(x + 50, ly - 10, 12, 12, 'neg cell')
    svg.text(x + 68, ly, '−1', 'fg', 12)
    svg.rect(x + 100, ly - 10, 12, 12, 'off cell')
    svg.text(x + 118, ly, '0', 'fg', 12)
    svg.text(x + 140, ly, 'each stored as a magnitude bit and a sign bit (set = negative)', 'muted', 11)
    y = ly + 30
    svg.text(30, y, 'Σ Ix² = popcount(magX) = %d' % sxx, 'fg', 13, mono=True)
    svg.text(30 + 2 * (pw + gap) - 40, y, 'Σ Iy² = popcount(magY) = %d' % syy, 'fg', 13, mono=True)
    col2 = 30 + 2 * (pw + gap) - 40
    svg.text(30, y + 22, 'Σ IxIy =   popcount(magX & magY & ~(signX ^ signY))', 'fg', 13, mono=True)
    svg.text(col2 + 120, y + 22, 'signs agree:  %d' % agree, 'muted', 13, mono=True)
    svg.text(30, y + 42, '        − popcount(magX & magY &  (signX ^ signY))', 'fg', 13, mono=True)
    svg.text(col2 + 120, y + 42, 'signs oppose: %d' % oppose, 'muted', 13, mono=True)
    svg.text(30, y + 62, '        = %d − %d = %d' % (agree, oppose, agree - oppose), 'fg', 13, mono=True)
    svg.text(30, y + 88, 'No multiply anywhere: the whole 2×2 gradient covariance is four population counts '
             'over masks.', 'muted', 11)
    return svg


def fig_padding():
    W, width, cell = 8, 13, 22
    words = 2
    total = words * W
    src = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1] + [0] * (total - width)
    notm = [1 - b for b in src]
    tail = [1 if i < width else 0 for i in range(total)]
    masked = [a & b for a, b in zip(notm, tail)]
    true_count = sum(1 - b for b in src[:width])

    svg = Svg(900, 214, 'A 13-pixel row in two 8-bit words; a word-wise NOT without the tail mask '
                        'sets the three padding bits, which a later count reads as pixels')
    gx, y0, dy = 190, 40, 50
    svg.text(gx, 24, 'width = %d, stored in %d words of %d bits' % (width, words, W), 'muted', 11)
    rows = [('source', src, None), ('NOT, whole words', notm, sum(notm)), ('NOT, then & tailMask', masked, sum(masked))]
    for k, (lab, bits, cnt) in enumerate(rows):
        y = y0 + k * dy

        def cls_of(i, b, k=k):
            if i >= width:
                return 'hi' if b else 'pad'
            return None
        svg.text(gx - 14, y + cell / 2 + 5, lab, 'fg', 13, 'end')
        bitrow(svg, gx, y, bits, cell, word=W, cls_of=cls_of)
        if cnt is not None:
            ok = cnt == true_count
            svg.text(gx + total * cell + 22, y + cell / 2 + 5,
                     'countNonZero = %d' % cnt + ('' if ok else ', %d of them padding' % (cnt - true_count)),
                     'fg' if ok else 'hi', 13, mono=True)
    svg.text(gx + width * cell + (total - width) * cell / 2, y0 - 6, 'padding', 'muted', 11, 'middle')
    svg.text(gx, y0 + 3 * dy + 8, 'A word-wise operation writes padding bits too. Every kernel that writes whole words '
             'masks the last one, so padding stays zero.', 'muted', 11)
    return svg


def fig_pyramid():
    src = ['11100001',
           '11000011',
           '01101111',
           '00111111']
    img = [[int(c) for c in r] for r in src]
    H, Wd = len(img), len(img[0])
    S = [[img[2 * y][2 * x] + img[2 * y][2 * x + 1] + img[2 * y + 1][2 * x] + img[2 * y + 1][2 * x + 1]
          for x in range(Wd // 2)] for y in range(H // 2)]
    NIn, NOut = 1, 3
    q = [[(s * ((1 << NOut) - 1) + 2 * ((1 << NIn) - 1)) // (4 * ((1 << NIn) - 1)) for s in r] for r in S]
    cell, oc = 26, 32
    svg = Svg(900, 262, 'A 2 by 2 box over a 1-bit image: each block counts 0 to 4, rescaled to a 3-bit value '
                        'stored as three planes')
    x0, y0 = 30, 50
    svg.text(x0, 26, '1-bit source', 'fg', 14, weight='600')
    grid(svg, x0, y0, img, cell)
    for by in range(H // 2):
        for bx in range(Wd // 2):
            svg.rect(x0 + 2 * bx * cell, y0 + 2 * by * cell, 2 * cell, 2 * cell, 'word')
    svg.rect(x0, y0, 2 * cell, 2 * cell, 'ring')

    x1 = x0 + Wd * cell + 70
    svg.arrow(x0 + Wd * cell + 12, y0 + H * cell / 2, x1 - 12, y0 + H * cell / 2)
    svg.text(x1, 26, 'count, 0–4', 'fg', 14, weight='600')
    for y, r in enumerate(S):
        for x, s in enumerate(r):
            svg.rect(x1 + x * oc, y0 + y * oc, oc, oc, ('hisoft' if (x, y) == (0, 0) else 'off') + ' cell')
            svg.text(x1 + x * oc + oc / 2, y0 + y * oc + oc / 2 + 5, str(s), 'fg', 14, 'middle', mono=True)

    x2 = x1 + (Wd // 2) * oc + 70
    svg.arrow(x1 + (Wd // 2) * oc + 12, y0 + oc, x2 - 12, y0 + oc)
    svg.text(x2, 26, '3-bit value', 'fg', 14, weight='600')
    for y, r in enumerate(q):
        for x, v in enumerate(r):
            svg.rect(x2 + x * oc, y0 + y * oc, oc, oc, ('hisoft' if (x, y) == (0, 0) else 'off') + ' cell')
            svg.text(x2 + x * oc + oc / 2, y0 + y * oc + oc / 2 + 5, str(v), 'fg', 14, 'middle', mono=True)

    x3 = x2 + (Wd // 2) * oc + 70
    svg.arrow(x2 + (Wd // 2) * oc + 12, y0 + oc, x3 - 12, y0 + oc)
    svg.text(x3, 26, 'stored as 3 planes', 'fg', 14, weight='600')
    pc = 18
    for p in range(NOut):
        yy = y0 + p * (2 * pc + 14)
        rows = [[(v >> p) & 1 for v in r] for r in q]
        grid(svg, x3, yy, rows, pc)
        svg.rect(x3, yy, pc, pc, 'ring')
        svg.text(x3 + (Wd // 2) * pc + 10, yy + pc + 4, 'plane %d' % p, 'muted', 11)

    ty = y0 + H * cell + 40
    svg.text(x0, ty, 'value = round(count × 7 / 4): the five counts 0, 1, 2, 3, 4 become 0, 2, 4, 5, 7, '
             'so white stays white at full scale.', 'fg', 12)
    svg.text(x0, ty + 22, 'The count is a bit-sliced sum of four 1-bit inputs, so it is word-parallel like the '
             'adder; the rescale is a comparison against constants.', 'muted', 11)
    svg.text(x0, ty + 40, 'Row pairs are read by index, so the vertical half of the downsample moves no bits; '
             'the horizontal half is a word-local unshuffle.', 'muted', 11)
    return svg


# ---------------------------------------------------------------------------------------
# Data charts


def report_table(path, heading):
    """The first markdown table under `heading` in a report, as a list of row dicts."""
    text = open(os.path.join(ROOT, path), encoding='utf-8').read()
    start = text.index('\n' + heading + '\n')
    lines = text[start:].split('\n')[2:]
    table = []
    for ln in lines:
        if ln.startswith('## '):
            break
        if ln.startswith('|'):
            table.append(ln)
        elif table:
            break
    cells = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in table]
    header, body = cells[0], cells[2:]
    return [dict(zip(header, r)) for r in body]


def fig_memory_chart():
    src = 'docs/reports/footprint.md'
    rows = report_table(src, '## Per operation')
    # The denoise row is left out on purpose: its ratio is the reference's seven-buffer
    # composition against a fused kernel, not the bit width, and the report says so.
    wanted = ['`erode` / `dilate`, 3×3',
              '`morphologyEx(MORPH_OPEN)`',
              'spatial derivative, both axes',
              '`goodFeaturesToTrack`, at the measured survivor count',
              'FAST input plane']
    names = {'`erode` / `dilate`, 3×3': 'erode / dilate, 3×3',
             '`morphologyEx(MORPH_OPEN)`': 'morphologyEx, MORPH_OPEN',
             'spatial derivative, both axes': 'derivative, both axes',
             '`goodFeaturesToTrack`, at the measured survivor count': 'goodFeaturesToTrack',
             'FAST input plane': 'FAST input plane'}
    by = {r['operation']: r for r in rows}
    missing = [w for w in wanted if w not in by]
    if missing:
        sys.exit('gen_writeup_figures: %s has no row %s' % (src, missing))

    def n(s):
        return int(s.replace(',', ''))

    data = [(names[w], n(by[w]['OpenCV, bytes']), n(by[w]['binCV, bytes']), by[w]['ratio']) for w in wanted]
    rh, top, lx, bw = 46, 50, 220, 450
    svg = Svg(860, top + len(data) * rh + 56, 'Peak working set per call at 640 by 480, binCV against OpenCV, '
                                             'each row drawn to its own OpenCV total')
    svg.text(lx, 24, 'Working set of one call, 640×480. Each row is scaled to its own OpenCV bar.', 'muted', 12)
    for k, (name, ocv, bcv, ratio) in enumerate(data):
        y = top + k * rh
        svg.text(lx - 14, y + 20, name, 'fg', 13, 'end')
        svg.rect(lx, y, bw, 14, 'bar1')
        svg.rect(lx, y + 17, max(2, bw * bcv / ocv), 14, 'bar2')
        svg.text(lx + bw + 10, y + 11, '{:,} B'.format(ocv), 'muted', 11, mono=True)
        svg.text(lx + bw * bcv / ocv + 8, y + 28, '{:,} B'.format(bcv), 'fg', 11, mono=True)
        svg.text(850, y + 20, ratio + ' smaller', 'fg', 13, 'end', weight='600')
    ly = top + len(data) * rh + 12
    svg.rect(lx, ly, 12, 12, 'bar1')
    svg.text(lx + 18, ly + 10, 'OpenCV, same content as CV_8U', 'fg', 12)
    svg.rect(lx + 230, ly, 12, 12, 'bar2')
    svg.text(lx + 248, ly + 10, 'binCV', 'fg', 12)
    svg.text(lx, ly + 34, 'Source: docs/reports/footprint.md, per operation. Exact, from buffer geometry; '
             'identical on x86-64 and aarch64.', 'muted', 11)
    return svg


# ---------------------------------------------------------------------------------------
# Architecture figures


def fig_register():
    """How many pixels one register or word holds, drawn to a common pixel scale."""
    svg = Svg(900, 270, 'An AVX2 register of bytes holds 32 pixels, as many as one uint32_t word of bits; '
                        'the same register holding bits holds 256')
    lx, x0, scale = 250, 270, 2.0
    rows = [('AVX2 register, one byte per pixel', 256, 32, 8, 'off', '32 pixels'),
            ('one uint32_t word, one bit per pixel', 32, 32, 1, 'on', '32 pixels'),
            ('AVX2 register, one bit per pixel', 256, 256, 1, 'on', '256 pixels')]
    y = 34
    for label, bits, pixels, bits_per_px, cls, note in rows:
        svg.text(lx, y + 17, label, 'fg', 13, 'end')
        w = bits * scale
        svg.rect(x0, y, w, 26, cls + ' cell')
        step = bits_per_px * scale
        if bits_per_px > 1:
            for i in range(1, pixels):
                svg.line(x0 + i * step, y, x0 + i * step, y + 26, 'edge')
        else:
            for i in range(32, bits, 32):
                svg.line(x0 + i * scale, y, x0 + i * scale, y + 26, 'edge')
        svg.rect(x0, y, w, 26, 'word')
        svg.text(x0 + w + 12, y + 17, note, 'fg', 13, weight='600')
        y += 58
    svg.text(x0, y - 10, 'bits →', 'muted', 11)
    svg.text(30, y + 18, 'Width drawn to scale in bits. Packing alone only matches a byte vector: a scalar word of bits '
             'holds exactly what an AVX2', 'muted', 11)
    svg.text(30, y + 34, 'register of bytes holds. The advantage appears when the bit-level logic also runs across '
             'the whole vector register.', 'muted', 11)
    return svg


def fig_popcount():
    svg = Svg(900, 330, 'Where the population count runs: in general registers on x86-64, and in the NEON '
                        'register file on aarch64, where a per-word count crosses between register files')

    def lane(y, label, sub, steps):
        svg.text(30, y - 14, label, 'fg', 14, weight='600')
        svg.text(30 + 8 * len(label) + 12, y - 14, sub, 'muted', 11)
        x = 30
        for i, (text, domain) in enumerate(steps):
            w = 16 + 7.4 * len(text)
            cls = {'gpr': 'off', 'vec': 'hisoft', 'cross': 'bg'}[domain]
            svg.rect(x, y, w, 30, cls + ' cell', rx=4)
            svg.text(x + w / 2, y + 20, text, 'fg', 12, 'middle', mono=True)
            if i + 1 < len(steps):
                nxt = steps[i + 1][1]
                hi = (domain == 'vec') != (nxt == 'vec')
                svg.arrow(x + w + 3, y + 15, x + w + 25, y + 15, hi=hi)
            x += w + 28
        return x

    lane(46, 'x86-64', 'POPCNT reads and writes general registers', [('load word', 'gpr'), ('popcnt', 'gpr'),
                                                                         ('add', 'gpr')])
    lane(124, 'aarch64, one word at a time', '', [('word in GPR', 'gpr'), ('fmov', 'gpr'), ('cnt', 'vec'),
                                                 ('addv', 'vec'), ('fmov', 'gpr'), ('add', 'gpr')])
    svg.text(740, 144, 'two crossings per word', 'hi', 12, weight='600')
    lane(202, 'aarch64, a whole window', '', [('ld1 128 bits', 'vec'), ('cnt', 'vec'), ('add bytes', 'vec'),
                                             ('… per row', 'vec'), ('widen, addv', 'vec'), ('fmov', 'gpr')])
    svg.text(740, 222, 'one crossing per window', 'fg', 12, weight='600')
    ly = 262
    svg.rect(30, ly, 14, 14, 'off cell', rx=3)
    svg.text(50, ly + 11, 'general-purpose registers', 'fg', 12)
    svg.rect(230, ly, 14, 14, 'hisoft cell', rx=3)
    svg.text(250, ly + 11, 'NEON register file', 'fg', 12)
    svg.line(400, ly + 7, 424, ly + 7, 'hiedge')
    svg.text(432, ly + 11, 'a crossing between the two', 'fg', 12)
    svg.text(30, ly + 40, 'aarch64 has no scalar population count, so the crossings cost about as much as the count. '
             'binCV offers counts over regions, masks', 'muted', 11)
    svg.text(30, ly + 56, 'and windows only, never over one word, so a kernel can keep the data in vector registers '
             'and pay the crossing once.', 'muted', 11)
    return svg


def fig_window():
    svg = Svg(900, 250, 'A 31-pixel window uses 31 of 32 bits in a uint32_t word and 31 of 64 in a uint64_t; '
                        'OpenCV processes 16 pixels per AVX2 operation on CV_16S')
    lx, x0, cell = 230, 250, 6.5
    rows = [('uint32_t word, one bit per pixel', 32, 31, '31 pixels per op, 97% used'),
            ('uint64_t word, one bit per pixel', 64, 31, '31 pixels per op, 48% used'),
            ('AVX2 register, CV_16S lanes', 16, 16, '16 pixels per op')]
    y = 30
    for k, (label, slots, used, note) in enumerate(rows):
        svg.text(lx, y + 15, label, 'fg', 13, 'end')
        slot_w = cell
        for i in range(slots):
            cls = 'on' if i < used else 'pad'
            svg.rect(x0 + i * slot_w, y, slot_w, 22, cls + ' cell')
        svg.rect(x0, y, slots * slot_w, 22, 'word')
        svg.text(x0 + slots * slot_w + 12, y + 15, note, 'fg', 12)
        y += 52
    svg.text(30, y + 8, 'The 31-pixel window row is the unit of work in the tracker. Matched to a 32-bit word it '
             'runs 31 pixels per operation against', 'muted', 11)
    svg.text(30, y + 24, 'OpenCV\'s 16: a 1.94× packing advantage, capped by the window size. A wider word '
             'lowers the utilisation instead of raising the cap.', 'muted', 11)
    return svg


def fig_narrowing():
    svg = Svg(900, 236, 'On a little-endian machine a 64-bit bit-plane row is byte-identical to a 32-bit '
                        'row with twice as many words, so it can be read as one without a copy')
    x0, bw = 150, 74
    y = 40
    svg.text(x0 - 14, y + 20, 'bytes in memory', 'fg', 13, 'end')
    for b in range(8):
        svg.rect(x0 + b * bw, y, bw, 30, 'off cell')
        svg.text(x0 + b * bw + bw / 2, y + 13, 'byte %d' % b, 'fg', 11, 'middle')
        svg.text(x0 + b * bw + bw / 2, y + 26, 'px %d–%d' % (8 * b, 8 * b + 7), 'muted', 10, 'middle', mono=True)
    y2 = y + 62
    svg.text(x0 - 14, y2 + 18, 'one uint64_t', 'fg', 13, 'end')
    svg.rect(x0, y2, 8 * bw, 28, 'on cell')
    svg.text(x0 + 4 * bw, y2 + 18, 'bits 0–63 = pixels 0–63', 'oninv', 12, 'middle', mono=True)
    y3 = y2 + 48
    svg.text(x0 - 14, y3 + 18, 'two uint32_t', 'fg', 13, 'end')
    for k in range(2):
        svg.rect(x0 + k * 4 * bw, y3, 4 * bw, 28, 'on cell')
        svg.rect(x0 + k * 4 * bw, y3, 4 * bw, 28, 'word')
        svg.text(x0 + (k * 4 + 2) * bw, y3 + 18, 'word %d: pixels %d–%d' % (k, 32 * k, 32 * k + 31), 'oninv', 12,
                 'middle', mono=True)
    svg.text(x0 + 8 * bw + 16, y2 + 18, 'stride S words', 'muted', 12)
    svg.text(x0 + 8 * bw + 16, y3 + 18, 'stride 2S words', 'muted', 12)
    svg.text(30, y3 + 56, 'Pixel x is bit x of the row either way, because the low byte comes first. '
             'narrowPlane reinterprets the view; nothing is copied or allocated.', 'muted', 11)
    return svg


def fig_ballot():
    svg = Svg(900, 230, 'On the GPU, each of the 32 threads in a warp tests one pixel, and __ballot_sync '
                        'returns all 32 answers as one packed word')
    x0, cw = 100, 22
    y = 36
    svg.text(x0 - 12, y, 'lane', 'muted', 11, 'end')
    pattern = [1 if (i * 7 + 3) % 5 < 2 else 0 for i in range(32)]
    for i in range(32):
        svg.text(x0 + i * cw + cw / 2, y, str(i), 'muted', 9, 'middle', mono=True)
        svg.rect(x0 + i * cw + 2, y + 6, cw - 4, 22, 'off cell', rx=3)
        svg.text(x0 + i * cw + cw / 2, y + 21, 'T' if pattern[i] else 'F', 'fg', 11, 'middle', mono=True)
    svg.text(x0 - 12, y + 21, 'pixel > t', 'muted', 11, 'end')
    y2 = y + 86
    svg.arrow(x0 + 16 * cw, y + 34, x0 + 16 * cw, y2 - 6)
    svg.text(x0 + 16 * cw + 10, y + 60, '__ballot_sync(mask, pixel > t): one instruction', 'fg', 12, mono=True)
    bitrow(svg, x0, y2, pattern, cw, word=32)
    svg.text(x0 - 12, y2 + 16, 'word', 'muted', 11, 'end')
    svg.text(30, y2 + 50, 'Lane i sets bit i, which is exactly where the format stores pixel i, so a warp packs a '
             '32-pixel word of the host\'s own layout', 'muted', 11)
    svg.text(30, y2 + 66, 'in one step. That is why the device word is uint32_t: it is both the integer width and the '
             'width of the hardware\'s own primitives.', 'muted', 11)
    return svg


def parse_ratio(cell):
    m = re.match(r'([0-9.]+)×', cell)
    return float(m.group(1)) if m else None


def fig_crossover_chart():
    src = 'docs/reports/limits.md'
    rows = report_table(src, '## 2. The crossover is real, and it moves with the architecture')
    pts = []
    for r in rows:
        m = re.match(r'box filter, (\d) → (\d)$', r['arm'])
        if m and m.group(1) == m.group(2):
            pts.append((int(m.group(1)), parse_ratio(r['x86-64 ratio']), parse_ratio(r['aarch64 ratio'])))
    if len(pts) < 5:
        sys.exit('gen_writeup_figures: %s crossover table not found or reshaped' % src)
    import math
    svg = Svg(860, 380, 'The box-filter downsample against cv::pyrDown across bit depths: binCV stays ahead '
                        'through 4 bits on the Cortex-A72 and is behind at every equal-width depth on x86-64')
    px0, px1, py0, py1 = 110, 600, 40, 300
    lo, hi = math.log10(0.05), math.log10(10)

    def Y(v):
        return py1 - (math.log10(v) - lo) / (hi - lo) * (py1 - py0)
    xs = {n: px0 + i * (px1 - px0) / (len(pts) - 1) for i, (n, _, _) in enumerate(pts)}
    for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10):
        svg.line(px0, Y(t), px1, Y(t), 'gridline')
        svg.text(px0 - 10, Y(t) + 4, ('%g×' % t), 'muted', 11, 'end')
    svg.line(px0, Y(1), px1, Y(1), 'refline')
    svg.text(px1 + 12, Y(1) - 6, 'binCV ahead ↑', 'fg', 11)
    svg.text(px1 + 12, Y(1) + 14, 'OpenCV ahead ↓', 'fg', 11)
    for n, x in xs.items():
        svg.text(x, py1 + 20, '%d → %d' % (n, n), 'fg', 12, 'middle', mono=True)
    svg.text((px0 + px1) / 2, py1 + 44, 'bits in → bits out', 'muted', 12, 'middle')
    svg.text(28, (py0 + py1) / 2, 'OpenCV time ÷ binCV time', 'muted', 12, 'middle')
    svg.parts[-1] = svg.parts[-1].replace('<text ', '<text transform="rotate(-90 28 %s)" ' % num((py0 + py1) / 2), 1)
    for idx, cls, dcls, name in ((2, 's1', 'd1', 'aarch64, Cortex-A72'), (1, 's2', 'd2', 'x86-64, Ryzen 5 5600X')):
        path = 'M' + ' L'.join('%s %s' % (num(xs[p[0]]), num(round(Y(p[idx]), 2))) for p in pts)
        svg.parts.append('<path d="%s" class="%s"/>' % (path, cls))
        for p in pts:
            x, y = xs[p[0]], Y(p[idx])
            svg.parts.append('<circle cx="%s" cy="%s" r="4" class="%s"/>' % (num(x), num(round(y, 2)), dcls))
            below = idx == 1 or 0.6 < p[idx] < 1.0
            first = p is pts[0]
            svg.text(x + (4 if first else 0), y + (19 if below else -10), '%g×' % p[idx], 'fg halo', 10,
                     'start' if first else 'middle', mono=True)
        last = pts[-1]
        svg.text(px1 + 62, Y(last[idx]) + 4, name, 'fg', 12, weight='600')
    svg.text(px0, 368, 'Source: docs/reports/limits.md, the crossover table. One thread each side; each machine has '
             'its own cv::pyrDown denominator.', 'muted', 11)
    return svg


def fig_lk_chart():
    src = 'docs/reports/limits.md'
    rows = report_table(src, '## 4. A footprint win is not a speed win')
    pts = [(r['frame'], float(r['input, KiB at 1 bit']), float(r['time, x86-64 (µs/point)']),
            float(r['time, aarch64 (µs/point)'])) for r in rows]
    svg = Svg(860, 330, 'Lucas-Kanade cost per point stays flat as the frame grows 36-fold at a fixed '
                        'point count, on both machines')
    px0, px1, py0, py1 = 110, 600, 36, 240
    lo, hi = 0.8, 1.2

    def Y(v):
        return py1 - (v - lo) / (hi - lo) * (py1 - py0)
    xs = [px0 + i * (px1 - px0) / (len(pts) - 1) for i in range(len(pts))]
    for t in (0.8, 0.9, 1.0, 1.1, 1.2):
        svg.line(px0, Y(t), px1, Y(t), 'refline' if t == 1.0 else 'gridline')
        svg.text(px0 - 10, Y(t) + 4, '%.1f×' % t, 'muted', 11, 'end')
    for x, p in zip(xs, pts):
        svg.text(x, py1 + 20, p[0], 'fg', 12, 'middle', mono=True)
        svg.text(x, py1 + 36, '%g KiB' % p[1], 'muted', 11, 'middle')
    svg.text(28, (py0 + py1) / 2, 'cost per point vs 320×240', 'muted', 12, 'middle')
    svg.parts[-1] = svg.parts[-1].replace('<text ', '<text transform="rotate(-90 28 %s)" ' % num((py0 + py1) / 2), 1)
    for idx, cls, dcls, name in ((3, 's1', 'd1', 'aarch64, Cortex-A72'), (2, 's2', 'd2', 'x86-64, Ryzen 5 5600X')):
        base = pts[0][idx]
        ys = [Y(p[idx] / base) for p in pts]
        svg.parts.append('<path d="M%s" class="%s"/>' % (' L'.join('%s %s' % (num(x), num(round(y, 2)))
                                                                    for x, y in zip(xs, ys)), cls))
        for x, y, p in zip(xs, ys, pts):
            svg.parts.append('<circle cx="%s" cy="%s" r="4" class="%s"/>' % (num(x), num(round(y, 2)), dcls))
            first = p is pts[0]
            svg.text(x + (4 if first else 0), y + (19 if idx == 2 else -10), '%g µs' % p[idx], 'fg halo', 10,
                     'start' if first else 'middle', mono=True)
        svg.text(px1 + 72, ys[-1] + (8 if idx == 2 else 0), name, 'fg', 12, weight='600')
    svg.text(px0, 300, '140 points, 31×31 window, one pyramid level. Labels are µs per point.', 'muted', 11)
    svg.text(px0, 316, 'Source: docs/reports/limits.md, the frame-size sweep.', 'muted', 11)
    return svg


def fig_cache_ladder():
    """Working sets of bitwiseAnd (two inputs and the output) against each machine's
    memory levels. The sizes are geometry, computed here; the cache sizes are the
    machines' own: docs/reports/README.md for the two application processors, ST's
    STM32H753 datasheet (DS12117) for the microcontroller."""
    import math
    machines = [
        ('Raspberry Pi 4', [('L1', 32768, '32 KiB'), ('L2', 1048576, '1 MiB'), ('DRAM', 1 << 27, '')]),
        ('x86-64 desktop', [('L1', 32768, '32 KiB'), ('L2', 524288, '512 KiB'), ('L3', 1 << 25, '32 MiB'),
                            ('DRAM', 1 << 27, '')]),
        ('Cortex-M7', [('L1', 16384, '16 KiB'), ('SRAM', 524288, '512 KiB'), ('none', 1 << 27, 'no larger RAM bank')]),
    ]
    sizes = [(640, 480), (1024, 1024), (8192, 4096)]
    x0, k = 250, 41

    def X(v):
        return x0 + math.log2(v / 4096) * k
    svg = Svg(900, 420, 'Working sets of bitwiseAnd as bits and as bytes at three image sizes, on a log scale '
                        'against the memory levels of a Raspberry Pi 4, an x86-64 desktop and a Cortex-M7')
    shade = {'L1': 'hisoft', 'L2': 'off', 'L3': 'pad', 'DRAM': 'bg', 'SRAM': 'off', 'none': 'pad'}
    for r, (name, bands) in enumerate(machines):
        y = 20 + r * 50
        svg.text(x0 - 12, y + 25, name, 'fg', 13, 'end', weight='600')
        prev = 4096
        for b, size, lab in bands:
            a, c = X(prev), X(size)
            svg.rect(a, y, c - a, 38, shade[b] + ' cell')
            label = lab if b == 'none' else b + ('  ' + lab if lab else '')
            svg.text(a + 6, y + 24, label, 'hi' if b == 'none' else 'fg', 12, weight='600')
            prev = size
    pi_l2 = X(1048576)
    svg.line(pi_l2, 14, pi_l2, 350, 'hiedge')
    svg.text(pi_l2 + 6, 186, 'Pi L2 ends', 'hi', 11)
    for i, (w, h) in enumerate(sizes):
        y = 220 + i * 52
        bits, byts = 3 * w * h // 8, 3 * w * h
        svg.text(x0 - 12, y + 5, '%d×%d' % (w, h), 'fg', 13, 'end')
        svg.line(X(bits), y, X(byts), y, 'edge')
        svg.parts.append('<circle cx="%s" cy="%s" r="8" class="on"/>' % (num(round(X(bits), 1)), y))
        svg.parts.append('<circle cx="%s" cy="%s" r="8" class="pad"/>' % (num(round(X(byts), 1)), y))
        svg.text(X(bits), y + 24, '{:,} B'.format(bits), 'muted', 10, 'middle', mono=True)
        svg.text(X(byts), y + 24, '{:,} B'.format(byts), 'muted', 10, 'middle', mono=True)
    svg.parts.append('<circle cx="%s" cy="372" r="7" class="on"/>' % x0)
    svg.text(x0 + 14, 376, 'binCV: two input planes and the output, as bits', 'fg', 12)
    svg.parts.append('<circle cx="%s" cy="396" r="7" class="pad"/>' % x0)
    svg.text(x0 + 14, 400, 'OpenCV: the same three frames as bytes', 'fg', 12)
    svg.text(890, 400, 'log scale, 4 KiB to 128 MiB', 'muted', 11, 'end')
    return svg


def fig_streaming():
    """The dense cost volume against binCV's streamed band. Shapes only: the
    figures that belong to it are measured and live in the report tables."""
    svg = Svg(900, 330, 'A dense stereo cost volume holds a cost for every pixel at every disparity; binCV '
                        'streams a band of rows through per-disparity running sums and emits the map row by row')
    svg.text(30, 26, 'Cost volume: every pixel at every disparity', 'fg', 14, weight='600')
    for k in range(10, -1, -1):
        svg.rect(40 + k * 14, 70 - k * 4 + 20, 220, 140, ('pad' if k else 'off') + ' cell')
    svg.text(40, 260, 'width × height × disparities, all resident', 'muted', 12)
    svg.text(40, 280, 'before any pixel picks its disparity', 'muted', 12)
    bx = 470
    svg.text(bx, 26, 'binCV: a band of rows, streamed', 'fg', 14, weight='600')
    svg.rect(bx, 50, 170, 200, 'off cell')
    svg.rect(bx, 120, 170, 36, 'on cell')
    svg.text(bx + 85, 143, 'window rows', 'oninv', 12, 'middle')
    svg.arrow(bx + 85, 172, bx + 85, 200)
    svg.text(bx + 85, 268, 'image pair, as bits', 'muted', 12, 'middle')
    rx = bx + 210
    svg.text(rx, 60, 'per disparity:', 'fg', 12)
    svg.text(rx, 78, 'a running window sum', 'fg', 12)
    for d in range(8):
        svg.rect(rx + d * 22, 92, 16, 46, 'hisoft cell')
    svg.text(rx, 160, 'add the entering row,', 'muted', 12)
    svg.text(rx, 177, 'drop the leaving one', 'muted', 12)
    svg.arrow(bx + 175, 138, rx - 6, 115)
    svg.rect(rx, 200, 176, 24, 'on cell')
    svg.text(rx + 88, 217, 'best disparity, this row', 'oninv', 11, 'middle')
    svg.arrow(rx + 168, 142, rx + 168, 196)
    svg.text(rx, 250, 'emitted, then the memory', 'muted', 12)
    svg.text(rx, 267, 'is reused for the next row', 'muted', 12)
    svg.text(30, 316, 'Nothing outlives a row except its answer, so the scratch depends on the width and the '
             'disparity range, not the image height.', 'muted', 11)
    return svg


FIGURES = {
    'premise-bytes-vs-bits.svg': fig_bytes_vs_bits,
    'premise-and.svg': fig_and,
    'premise-shift-or.svg': fig_shift_or,
    'premise-bitplanes.svg': fig_bitplanes,
    'premise-adder.svg': fig_adder,
    'premise-ternary-covariance.svg': fig_ternary,
    'premise-padding.svg': fig_padding,
    'premise-pyramid.svg': fig_pyramid,
    'premise-memory.svg': fig_memory_chart,
    'architectures-register.svg': fig_register,
    'architectures-popcount.svg': fig_popcount,
    'architectures-window.svg': fig_window,
    'architectures-narrowing.svg': fig_narrowing,
    'architectures-ballot.svg': fig_ballot,
    'architectures-crossover.svg': fig_crossover_chart,
    'architectures-lk-frame-size.svg': fig_lk_chart,
    'architectures-cache.svg': fig_cache_ladder,
    'premise-streaming.svg': fig_streaming,
}


def main():
    check = '--check' in sys.argv[1:]
    stale = []
    os.makedirs(OUT, exist_ok=True)
    for name, fn in FIGURES.items():
        path = os.path.join(OUT, name)
        want = fn().render()
        have = open(path, encoding='utf-8').read() if os.path.exists(path) else None
        if have == want:
            continue
        if check:
            stale.append(name)
        else:
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(want)
            print('wrote', os.path.relpath(path, ROOT))
    if check:
        if stale:
            print('out of date: %s\nrun: python3 scripts/gen_writeup_figures.py' % ', '.join(stale))
            return 1
        print('writeup figures: %d up to date' % len(FIGURES))
    return 0


if __name__ == '__main__':
    sys.exit(main())
