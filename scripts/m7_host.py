#!/usr/bin/env python3
"""m7_host.py -- drive the STM32H753 firmware from WSL: flash it, talk to it, log it.

WHY IT EXISTS.  Every figure the Cortex-M7 publishes has to come from a run a
reader can repeat and a gate can check, the same as a figure from the Pi.  The
firmware in targets/stm32h753/ is a command loop (see its main.cpp); this is the
other end of that loop, plus the two things around it that are awkward from WSL:

  * FLASHING through the ST-LINK's mass-storage drive.  Windows caches that
    drive's tiny virtual FAT and then reports a copy onto it as "corrupted" --
    the fix is to eject the volume, let it re-enumerate, and copy from a
    Windows-local path.  No debugger, no driver, no administrator.
  * THE SERIAL PORT, which the ST-LINK presents to Windows as COM3.  usbipd can
    pass the whole device into WSL, but its service has crashed on this host
    twice and only an administrator can restart it.  So scripts/m7_bridge.ps1
    opens the port on the Windows side and connects to a socket this script
    listens on; WSL2 forwards Windows' localhost to it.

USAGE
    scripts/m7_host.py flash  IMAGE                  e.g. bincv_m7_reduce
    scripts/m7_host.py run    CMD [CMD ...]          one session, print replies
    scripts/m7_host.py link   [--bytes N] [--repeats R]   link integrity + rate
    scripts/m7_host.py launches -n N -o LOG --image IMAGE [--build DIR] CMD [CMD ...]
    scripts/m7_host.py track --blob BSQ --reference REF -o LOG [--frames N]

There is one firmware image per benchmark (targets/stm32h753/CMakeLists.txt says
why). `track` runs the feature tracker image (bincv_m7_track) over a BSQ1 blob, serving
the rows it asks for, and holds its per-frame output to REF -- what
benchmark/frontend_sequence --out wrote for the same blob on the host: every
frame's integer digest exactly, and Lucas–Kanade's points bit for bit, with the
largest difference reported where they are not.

`launches` is run_launches.sh for the board: it builds IMAGE from the
current tree, flashes it, then takes N launches -- each one a system reset, so
caches, stack and every static start cold -- and runs the commands in each.
The log carries the same header contract (`# benchmark`, `# commit`,
`# sources`) that check_figure_staleness.py reads, so a figure taken here goes
stale when the code under it moves, like any other.

EXIT CODES
    0   done
    1   the board answered ERR, a transfer failed its CRC, or a launch failed
    2   bad invocation, or the board / its drive could not be found
"""

import argparse
import datetime
import hashlib
import os
import socket
import subprocess
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUILD = os.path.join(ROOT, 'build-m7')
IMAGE_DIR = os.path.join('targets', 'stm32h753')


def image_rel(name):
    return os.path.join(IMAGE_DIR, name + '.bin')

BRIDGE = os.path.join(ROOT, 'scripts', 'm7_bridge.ps1')
VOLUME_LABEL = 'NOD_H753ZI'
DEFAULT_PORT = 'COM3'
DEFAULT_BAUD = 4000000
TCP_PORT = 47321


def die(msg, code=2):
    print(f'm7_host.py: {msg}', file=sys.stderr)
    sys.exit(code)


def powershell(command, timeout=60):
    r = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def winpath(path):
    return subprocess.run(['wslpath', '-w', path], capture_output=True, text=True,
                          check=True).stdout.strip()


# ----------------------------------------------------------------------- flashing

def board_drive():
    rc, out, _ = powershell(
        f"(Get-Volume | Where-Object FileSystemLabel -eq '{VOLUME_LABEL}').DriveLetter")
    letter = out.strip()
    return f'{letter}:' if rc == 0 and len(letter) == 1 else None


def flash(image):
    """Copy `image` onto the ST-LINK's drive, the way that works from WSL.

    Success is the .bin disappearing from the drive with no FAIL.TXT beside it:
    the ST-LINK consumes the file, programs the part and resets it.
    """
    if not os.path.isfile(image):
        die(f'no image at {image}')
    drive = board_drive()
    if drive is None:
        die(f'no volume labelled {VOLUME_LABEL}: is the board connected, and not attached '
            'to WSL through usbipd?')
    rc, tmp, _ = powershell('$env:TEMP')
    if rc != 0 or not tmp:
        die('could not read the Windows temp directory')
    name = os.path.basename(image)
    staged_win = tmp + '\\' + name
    rc, staged_wsl, _ = 0, subprocess.run(['wslpath', '-u', staged_win], capture_output=True,
                                          text=True, check=True).stdout.strip(), ''
    with open(image, 'rb') as src, open(staged_wsl, 'wb') as dst:
        dst.write(src.read())

    script = f"""
$ErrorActionPreference = 'Stop'
(New-Object -ComObject Shell.Application).Namespace(17).ParseName('{drive}\\').InvokeVerb('Eject')
$deadline = (Get-Date).AddSeconds(30)
Start-Sleep -Seconds 3
while (-not (Get-Volume | Where-Object FileSystemLabel -eq '{VOLUME_LABEL}')) {{
    if ((Get-Date) -gt $deadline) {{ throw 'the drive did not come back after eject' }}
    Start-Sleep -Milliseconds 500
}}
$d = (Get-Volume | Where-Object FileSystemLabel -eq '{VOLUME_LABEL}').DriveLetter + ':'
Start-Sleep -Seconds 2
Copy-Item -LiteralPath '{staged_win}' -Destination "$d\\{name}" -Force
$deadline = (Get-Date).AddSeconds(30)
while (Test-Path "$d\\{name}") {{
    if ((Get-Date) -gt $deadline) {{ throw 'the image was not consumed' }}
    Start-Sleep -Milliseconds 500
}}
Start-Sleep -Seconds 2
$d = (Get-Volume | Where-Object FileSystemLabel -eq '{VOLUME_LABEL}').DriveLetter + ':'
if (Test-Path "$d\\FAIL.TXT") {{ Get-Content "$d\\FAIL.TXT"; throw 'the ST-LINK wrote FAIL.TXT' }}
'flashed'
"""
    rc, out, err = powershell(script, timeout=120)
    if rc != 0 or 'flashed' not in out:
        die(f'flash failed: {out} {err}', 1)
    print(f'flashed {os.path.relpath(image, ROOT)} ({os.path.getsize(image)} bytes)', file=sys.stderr)


# ------------------------------------------------------------------------- session

class Session:
    """One bridged connection to the board's serial port."""

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, timeout=60.0):
        self.timeout = timeout
        lst = socket.socket()
        lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lst.bind(('127.0.0.1', TCP_PORT))
        lst.listen(1)
        lst.settimeout(60)
        self.bridge = subprocess.Popen(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
             '-File', winpath(BRIDGE), '-Port', port, '-Baud', str(baud),
             '-TcpPort', str(TCP_PORT)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.sock, _ = lst.accept()
        except socket.timeout:
            self.bridge.kill()
            out, err = self.bridge.communicate()
            die(f'the serial bridge never connected: {out.decode(errors="replace")} '
                f'{err.decode(errors="replace")}')
        finally:
            lst.close()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b''

    def close(self):
        try:
            self.sock.close()
        finally:
            try:
                self.bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.bridge.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def write(self, data):
        self.sock.sendall(data)

    def write_line(self, line):
        self.write(line.encode() + b'\n')

    def read_line(self, timeout=None):
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while b'\n' not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f'no line from the board within the timeout '
                                   f'(partial: {self.buf[:80]!r})')
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError('the bridge closed the connection')
            self.buf += chunk
        line, self.buf = self.buf.split(b'\n', 1)
        return line.rstrip(b'\r').decode(errors='replace')

    def until_done(self, timeout=None, echo=None):
        """Lines up to and including the OK/ERR line that ends every reply."""
        lines = []
        while True:
            line = self.read_line(timeout)
            lines.append(line)
            if echo:
                echo(line)
            if line.startswith('OK') or line.startswith('ERR'):
                return lines

    def command(self, cmd, timeout=None, echo=None):
        self.write_line(cmd)
        return self.until_done(timeout, echo)

    def sync(self, attempts=20):
        """Get to a known state: send PING until a HELLO comes back, discarding
        anything the board printed before anyone was listening."""
        for _ in range(attempts):
            self.write_line('')
            self.write_line('PING')
            try:
                while True:
                    line = self.read_line(timeout=1.0)
                    if line.startswith('HELLO'):
                        self.until_done(timeout=5)
                        return line
            except TimeoutError:
                continue
        die('the board never answered PING', 1)

    def wait_boot(self, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.read_line(timeout=max(0.1, deadline - time.monotonic()))
            except TimeoutError:
                break
            if line.startswith('BOOT') or line.startswith('FATAL'):
                return line
        return None


# ------------------------------------------------------------------- subcommands

def cmd_run(args):
    rc = 0
    with Session(args.port, args.baud) as s:
        print(s.sync())
        for c in args.commands:
            lines = s.command(c, timeout=args.timeout, echo=print)
            if lines[-1].startswith('ERR'):
                rc = 1
    return rc


def cmd_link(args):
    """Send random bytes through ECHO and hold the board to their CRC."""
    rc = 0
    with Session(args.port, args.baud) as s:
        print(s.sync())
        rng = os.urandom
        for r in range(args.repeats):
            data = rng(args.bytes)
            s.write_line(f'ECHO {args.bytes}')
            ready = s.read_line()
            if not ready.startswith('SEND'):
                print(ready)
                return 1
            t0 = time.monotonic()
            s.write(data + zlib.crc32(data).to_bytes(4, 'little'))
            lines = s.until_done()
            dt = time.monotonic() - t0
            for line in lines:
                print(line)
            if lines[-1].startswith('ERR'):
                rc = 1
            print(f'  host round trip {dt:.3f} s, {args.bytes / dt / 1024:.1f} KiB/s')
    return rc


def compiler_line(build):
    """The cross compiler's own version line. A toolchain file sets the compiler
    without caching it, so it is read from the file CMake writes after probing."""
    import glob
    cxx = None
    for probe in glob.glob(os.path.join(build, 'CMakeFiles', '*', 'CMakeCXXCompiler.cmake')):
        for line in open(probe):
            if line.startswith('set(CMAKE_CXX_COMPILER '):
                cxx = line.split('"')[1]
                break
    if not cxx:
        return '(unknown)'
    r = subprocess.run([cxx, '--version'], capture_output=True, text=True)
    return r.stdout.splitlines()[0] if r.stdout else '(unknown)'


def git_commit():
    sha = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', ROOT, 'diff', '--quiet']).returncode != 0
    return sha + (' (dirty)' if dirty else '')


def sources_line(image):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'check_figure_staleness.py'),
                        '--stamp', image], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else \
        f'(unmappable -- the gate could not resolve {image}; see check_figure_staleness.py --explain)'


def cmd_launches(args):
    build = os.path.abspath(args.build)
    r = subprocess.run(['cmake', '--build', build, '--target', args.image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
        die('the image did not build', 1)
    image = os.path.join(build, image_rel(args.image))
    if not args.no_flash:
        flash(image)
    image_hash = hashlib.sha256(open(image, 'rb').read()).hexdigest()[:12]

    failed = 0
    with Session(args.port, args.baud) as s, open(args.output, 'w') as log:
        hello = s.sync()
        clock = s.command('CLOCK', timeout=15)
        log.write(f'# launch sweep -- {args.n} system resets of one firmware image\n')
        log.write(f'# benchmark: {args.image} {" ".join(args.commands)}\n')
        log.write(f'# launches: {args.n}\n')
        log.write('# arch:     cortex-m7\n')
        log.write('# host:     NUCLEO-H753ZI (STM32H753ZI)\n')
        log.write(f'# compiler: {compiler_line(build)}\n')
        log.write(f'# commit:   {git_commit()}\n')
        log.write(f'# sources:  {sources_line(args.image)}\n')
        log.write(f'# image:    {image_rel(args.image)} sha256 {image_hash}\n')
        log.write(f'# board:    {hello}\n')
        log.write(f'# clock:    {clock[0]}\n')
        log.write(f'# date:     {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}\n')
        log.write('# Each launch is a system reset: caches, stack and statics start cold.\n')
        log.write('# NO launch is discarded.\n\n')
        log.flush()
        for i in range(1, args.n + 1):
            log.write(f'### run {i}\n')
            s.write_line('RESET')
            boot = s.wait_boot()
            if boot is None or boot.startswith('FATAL'):
                log.write(f'### run {i} FAILED: no boot ({boot})\n')
                failed = 1
                s.sync()
                continue
            s.sync()
            # The core runs from the ST-LINK's 8 MHz clock, which wanders by a few
            # hundred ppm; every launch records what it was.
            log.write('\n'.join(s.command('CLOCK', timeout=15)) + '\n')
            for c in args.commands:
                lines = s.command(c, timeout=args.timeout)
                log.write('\n'.join(lines) + '\n')
                if lines[-1].startswith('ERR'):
                    failed = 1
            log.flush()
            print(f'\r  launch {i}/{args.n}', end='', file=sys.stderr)
        print(file=sys.stderr)
        log.write(f'\n# date:     {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}\n')
        log.write('# done\n')
    print(f'wrote {args.output} ({args.n} launches)', file=sys.stderr)
    return failed


def read_frame_lines(path):
    """F and P lines by frame number, from a reference or a board run."""
    f_lines, p_lines = {}, {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('F '):
                f_lines[int(line.split()[1])] = line.strip()
            elif line.startswith('P '):
                p_lines[int(line.split()[1])] = line.strip()
    return f_lines, p_lines


def decode_points(p_line):
    import struct as st
    out = []
    for tok in p_line.split()[2:]:
        x = st.unpack('<f', bytes.fromhex(tok[0:8])[::-1])[0]
        y = st.unpack('<f', bytes.fromhex(tok[8:16])[::-1])[0]
        out.append((x, y, tok[16] == '1'))
    return out


def compare_runs(board_path, ref_path):
    """What the board did against what the host reference did, frame by frame."""
    bf, bp = read_frame_lines(board_path)
    rf, rp = read_frame_lines(ref_path)
    frames = sorted(bf)
    f_match = sum(1 for f in frames if rf.get(f) == bf[f])
    p_exact = sum(1 for f in frames if rp.get(f) == bp.get(f))
    pts = differ = status_differ = 0
    max_d = 0.0
    for f in frames:
        if rp.get(f) == bp.get(f) or f not in rp or f not in bp:
            if f in bp:
                pts += len(bp[f].split()) - 2
            continue
        a, b = decode_points(bp[f]), decode_points(rp[f])
        for (x0, y0, s0), (x1, y1, s1) in zip(a, b):
            pts += 1
            if s0 != s1:
                status_differ += 1
            if (x0, y0) != (x1, y1):
                differ += 1
                max_d = max(max_d, ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5)
    first_bad = next((f for f in frames if rf.get(f) != bf[f]), None)
    return {
        'frames': len(frames), 'f_match': f_match, 'p_exact': p_exact, 'points': pts,
        'points_differ': differ, 'status_differ': status_differ, 'max_diff_px': max_d,
        'first_digest_mismatch': first_bad,
    }


def cmd_track(args):
    import struct as st
    build = os.path.abspath(args.build)
    image_name = 'bincv_m7_track'
    r = subprocess.run(['cmake', '--build', build, '--target', image_name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
        die('the image did not build', 1)
    image = os.path.join(build, image_rel(image_name))
    if not args.no_flash:
        flash(image)
    image_hash = hashlib.sha256(open(image, 'rb').read()).hexdigest()[:12]

    blob = open(args.blob, 'rb')
    magic, mode, w, h, count = st.unpack('<4sIIII', blob.read(20))
    if magic != b'BSQ1' or mode != 0:
        die(f'{args.blob} is not an 8-bit BSQ1 blob')
    frames = min(count, args.frames) if args.frames else count
    side = args.frames_out or (args.output + '.frames')

    with Session(args.port, args.baud) as s, open(args.output, 'w') as log, \
            open(side, 'w') as per_frame:
        hello = s.sync()
        clock = s.command('CLOCK', timeout=15)
        settings = []
        for kv in args.set:
            reply = s.command('SET ' + kv.replace('=', ' '), timeout=10)
            if reply[-1].startswith('ERR'):
                die(f'the board refused SET {kv}: {reply[-1]}', 1)
            settings.append(reply[0])
        log.write('# feature tracker on the board -- one run of a firmware image over a sequence\n')
        log.write(f'# benchmark: {image_name} TRACK {frames} {w} {h}\n')
        log.write('# launches: 1\n')
        log.write('# arch:     cortex-m7\n')
        log.write('# host:     NUCLEO-H753ZI (STM32H753ZI)\n')
        log.write(f'# compiler: {compiler_line(build)}\n')
        log.write(f'# commit:   {git_commit()}\n')
        log.write(f'# sources:  {sources_line(image_name)}\n')
        log.write(f'# image:    {image_rel(image_name)} sha256 {image_hash}\n')
        log.write(f'# board:    {hello}\n')
        log.write(f'# clock:    {clock[0]}\n')
        log.write(f'# sequence: {os.path.basename(args.blob)}, {frames} frames of {w}x{h}\n')
        log.write(f'# reference: {os.path.basename(args.reference)}\n')
        log.write(f'# settings: {" ".join(settings) if settings else "(image defaults)"}\n')
        log.write(f'# date:     {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}\n')
        log.write(f'# per-frame F/P/T lines: {os.path.basename(side)} (not committed: ~3 KB a frame)\n\n')
        log.flush()

        t_start = time.monotonic()
        s.write_line(f'TRACK {frames} {w} {h}')
        while True:
            line = s.read_line(timeout=300)
            if line.startswith('NEED '):
                _, f, row, n = line.split()
                f, row, n = int(f), int(row), int(n)
                blob.seek(32 + f * w * h + row * w)
                data = blob.read(n * w)
                s.write(data + zlib.crc32(data).to_bytes(4, 'little'))
                if row == 0 and f % 50 == 0:
                    print(f'\r  frame {f}/{frames}', end='', file=sys.stderr)
            elif line[:2] in ('F ', 'P ', 'T '):
                per_frame.write(line + '\n')
            else:
                log.write(line + '\n')
                if line.startswith('OK') or line.startswith('ERR'):
                    break
        print(file=sys.stderr)
        per_frame.flush()
        took = time.monotonic() - t_start

        c = compare_runs(side, args.reference)
        log.write('\n-- against the host reference --\n')
        log.write(f"frames compared          : {c['frames']}\n")
        log.write(f"integer digests equal    : {c['f_match']} of {c['frames']}\n")
        log.write(f"points bit-identical     : {c['p_exact']} of {c['frames']} frames\n")
        log.write(f"points that differ       : {c['points_differ']} of {c['points']}, "
                  f"largest {c['max_diff_px']:.3g} px, status differs on {c['status_differ']}\n")
        if c['first_digest_mismatch'] is not None:
            log.write(f"first digest mismatch    : frame {c['first_digest_mismatch']}\n")
        log.write(f'\n# wall time {took:.0f} s, mostly the link\n# done\n')
    text = open(args.output).read()
    if '-- per frame' in text:
        print('-- per frame' + text.split('-- per frame', 1)[1], file=sys.stderr)
    ok = c['f_match'] == c['frames']
    print(f'wrote {args.output}; digests equal on {c["f_match"]} of {c["frames"]} frames',
          file=sys.stderr)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', default=DEFAULT_PORT)
    p.add_argument('--baud', type=int, default=DEFAULT_BAUD)
    sub = p.add_subparsers(dest='sub', required=True)

    f = sub.add_parser('flash')
    f.add_argument('image', help='an image name (bincv_m7_reduce) or a path to a .bin')
    f.add_argument('--build', default=DEFAULT_BUILD)

    r = sub.add_parser('run')
    r.add_argument('--timeout', type=float, default=120.0)
    r.add_argument('commands', nargs='+')

    l = sub.add_parser('link')
    l.add_argument('--bytes', type=int, default=32768)
    l.add_argument('--repeats', type=int, default=3)

    n = sub.add_parser('launches')
    n.add_argument('-n', type=int, required=True,
                   help='launches; required and without a default, as in run_launches.sh')
    n.add_argument('-o', '--output', required=True)
    n.add_argument('--image', required=True, help='e.g. bincv_m7_reduce')
    n.add_argument('--build', default=DEFAULT_BUILD)
    n.add_argument('--no-flash', action='store_true',
                   help='the image on the board is already the one just built')
    n.add_argument('--timeout', type=float, default=600.0)
    n.add_argument('commands', nargs='+')

    t = sub.add_parser('track')
    t.add_argument('--blob', required=True)
    t.add_argument('--reference', required=True,
                   help='benchmark/frontend_sequence --out over the same blob and frames')
    t.add_argument('-o', '--output', required=True)
    t.add_argument('--frames', type=int, default=0)
    t.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                   help="a board setting before the run, e.g. lkwc=0 or rt=0")
    t.add_argument('--frames-out', default='',
                   help="where the per-frame lines go (default: beside the log)")
    t.add_argument('--build', default=DEFAULT_BUILD)
    t.add_argument('--no-flash', action='store_true')

    args = p.parse_args()
    if args.sub == 'track':
        return cmd_track(args)
    if args.sub == 'flash':
        path = args.image if args.image.endswith('.bin') else \
            os.path.join(args.build, image_rel(args.image))
        flash(os.path.abspath(path))
        return 0
    if args.sub == 'run':
        return cmd_run(args)
    if args.sub == 'link':
        return cmd_link(args)
    if args.sub == 'launches':
        return cmd_launches(args)
    return 2


if __name__ == '__main__':
    sys.exit(main())
