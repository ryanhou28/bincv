#!/usr/bin/env python3
"""Turn a directory of frames into one flat sequence blob ("BSQ1").

This is the HOST half of the no-codec decision (ARCHITECTURE 7): binCV links no
decoder on any target, so the decode happens here, where the decoders live, and
what travels to the target is one flat file -- a 32-byte header, then frames
back to back -- that works unchanged as an fread/mmap file, an app asset, an
`xxd -i` array in flash, or a stream over USB/UART. The reader is
include/bincv/io/sequence.hpp, and the format is documented there; the u32
header fields are little-endian, which is every target binCV runs on.

Two modes, because they test different things:

  --mode 8bit    body per frame = a P5 body (width*height bytes, row-major).
                 ~353 KB per 752x480 frame; exercises the WHOLE pipeline,
                 sensor stage and packing included.
  --mode packed  body per frame = a P4 body (rows padded to a byte boundary,
                 MSB first). ~44 KB per 752x480 frame -- 8x more frames in the
                 same flash -- and the sensor stage runs HERE, so the blob
                 tests only what is downstream of the binary frame.

The packed mode's sensor stage is the reference pipeline's two-stage
preprocessing, exactly as benchmark/frontend_sequence.cpp spells it (and as
binCV's own medianWide + edgeThreshold reproduce bit for bit):

  1. the L-shaped three-pixel median -- min/max over {above, center, right},
     out-of-range neighbours reading as ZERO;
  2. |d/dx| and |d/dy| over [-1, 0, 1] (filter2D, BORDER_REFLECT_101),
     edge = (|dx| >= threshold) OR (|dy| >= threshold).

With cv2 importable the filter is cv2.filter2D, the same call the reference
makes; without it, a numpy fallback pads with np.pad(mode="reflect"), which is
exactly BORDER_REFLECT_101 (the edge sample is NOT repeated). The two produce
identical bits; tests/test_sequence.cpp pins the shipped fixtures against
binCV's own spelling of the same stage, so the agreement is enforced without
python at test time.

Frames are read from *.png and *.pgm in the input directory, sorted by
filename. Loading uses cv2 where importable, else PIL. For GRAYSCALE sources
the two agree byte for byte; a color source is converted with the loader's own
luma rounding, which differs between the two -- keep dataset frames grayscale.

Usage:
  scripts/make_sequence_blob.py FRAME_DIR -o out.bsq --mode 8bit
  scripts/make_sequence_blob.py FRAME_DIR -o out.bsq --mode packed --threshold 17 --max-frames 100
"""

import argparse
import pathlib
import struct
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

MAGIC = b"BSQ1"
MODE_8BIT = 0
MODE_PACKED = 1


def load_gray(path):
    """One frame as a 2-D uint8 array, or None if unreadable."""
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def reference_denoise(img):
    """The reference's three-pixel median: min/max over {above, center, right}.

    A port of benchmark/frontend_sequence.cpp's referenceDenoise, slicing
    included: the shifted neighbours are built as ZEROS and the overlapping
    region copied in, so the row and column that fall off the edge keep the
    zeros -- that IS the border rule, specified by what the copies do not write.
    """
    right = np.zeros_like(img)
    above = np.zeros_like(img)
    right[:, :-1] = img[:, 1:]
    above[1:, :] = img[:-1, :]
    if cv2 is not None:
        a = cv2.min(above, img)
        b = cv2.max(above, img)
        c = cv2.min(b, right)
        return cv2.max(a, c)
    a = np.minimum(above, img)
    b = np.maximum(above, img)
    c = np.minimum(b, right)
    return np.maximum(a, c)


def reference_edge_filter(gray, threshold):
    """The reference's edge mask: |d/dx| >= t OR |d/dy| >= t over [-1, 0, 1].

    filter2D CORRELATES (dst(x) = src(x+1) - src(x-1)) and defaults to
    BORDER_REFLECT_101; the numpy arm reproduces both -- np.pad(mode="reflect")
    is reflect-101 (the edge sample is not repeated), and the slicing below is
    the same correlation. uint8 differences are exact in float32 and in int16
    alike, so the two arms agree bit for bit.
    """
    if cv2 is not None:
        kx = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        ky = kx.reshape(3, 1)
        dx = cv2.filter2D(gray, cv2.CV_32F, kx)
        dy = cv2.filter2D(gray, cv2.CV_32F, ky)
    else:
        g = gray.astype(np.int16)
        px = np.pad(g, ((0, 0), (1, 1)), mode="reflect")
        py = np.pad(g, ((1, 1), (0, 0)), mode="reflect")
        dx = px[:, 2:] - px[:, :-2]
        dy = py[2:, :] - py[:-2, :]
    return (np.abs(dx) >= threshold) | (np.abs(dy) >= threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Pack a directory of frames into one BSQ1 sequence blob.")
    parser.add_argument("frame_dir", type=pathlib.Path,
                        help="directory of *.png / *.pgm frames, ordered by filename")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="output blob path")
    parser.add_argument("--mode", choices=("8bit", "packed"), required=True,
                        help="8bit: P5-shaped bodies; packed: sensor stage here, P4-shaped bodies")
    parser.add_argument("--threshold", type=int, default=17,
                        help="packed mode's edge threshold (default 17, the reference's)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = all)")
    args = parser.parse_args()

    files = sorted(p for p in args.frame_dir.iterdir()
                   if p.suffix.lower() in (".png", ".pgm"))
    if args.max_frames > 0:
        files = files[:args.max_frames]
    if not files:
        sys.exit(f"error: no .png or .pgm frames in {args.frame_dir}")

    mode = MODE_8BIT if args.mode == "8bit" else MODE_PACKED
    width = height = None
    bodies = []
    for path in files:
        gray = load_gray(path)
        if gray is None:
            sys.exit(f"error: cannot read {path}")
        if width is None:
            height, width = gray.shape
        elif gray.shape != (height, width):
            sys.exit(f"error: {path} is {gray.shape[1]}x{gray.shape[0]}, "
                     f"expected {width}x{height} -- one blob, one frame size")
        if mode == MODE_8BIT:
            bodies.append(np.ascontiguousarray(gray).tobytes())
        else:
            edge = reference_edge_filter(reference_denoise(gray), args.threshold)
            # A P4 body: MSB-first within each byte, every row padded to a byte
            # boundary with zeros -- np.packbits along axis 1 is exactly that.
            bodies.append(np.packbits(edge, axis=1).tobytes())

    header = struct.pack("<4sIIII", MAGIC, mode, width, height, len(bodies))
    header += b"\x00" * (32 - len(header))
    with open(args.output, "wb") as out:
        out.write(header)
        for body in bodies:
            out.write(body)

    per_frame = len(bodies[0])
    total = 32 + per_frame * len(bodies)
    print(f"wrote {args.output}")
    print(f"  mode        {args.mode} ({mode})"
          + (f", threshold {args.threshold}" if mode == MODE_PACKED else ""))
    print(f"  frames      {len(bodies)} of {width}x{height}")
    print(f"  bytes/frame {per_frame}")
    print(f"  total       {total} bytes (32-byte header + bodies)")


if __name__ == "__main__":
    main()
