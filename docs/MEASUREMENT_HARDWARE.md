# Setting Up the Reference Measurement Device

One-time setup for the Cortex-A device that closes binCV's performance
experiments. Nothing in the normal development loop depends on this — see
[What works without it](#what-works-without-it).

**Why a separate device:** E-1, E-2 and E-3 are cache-hierarchy questions, and a
laptop hides the effect they measure. A Pi 4's Cortex-A72 has 32 KiB L1D and 1 MiB
shared L2, against roughly 128 KiB and 12 MiB on an M-series core. Full rationale
in [EXPERIMENTS.md § Measurement platforms](../EXPERIMENTS.md#measurement-platforms).

---

## Hardware

| Item | Requirement | Why |
|---|---|---|
| Raspberry Pi 4 | any RAM size | Cortex-A72 is the target class |
| **Official 5.1V 3A PSU** | **not a PC USB port** | see below — this one matters |
| Ethernet cable | to the router | fast `rsync`, no WiFi jitter |
| microSD | 16 GB+ | — |
| Heatsink or fan | strongly recommended | prevents throttled (invalid) runs |

### Do not power it from a PC USB port

Pi 4 supports USB-C gadget mode, so "just plug it into the PC" looks attractive.
It fails for this use case specifically:

| Source | Supplies |
|---|---|
| Pi 4 under load | ~1.2–1.5 A |
| Official PSU | 3 A — fine |
| USB-A 3.0 port | 0.9 A — **insufficient** |
| USB-C without PD negotiation | typically 1.5 A — **marginal** |

The Pi 4's USB-C port is its power port. Underpowered, the firmware sets the
undervoltage bit — the same `vcgencmd get_throttled` flag the runner script uses to
mark a run **INVALID**. The result is a rig that intermittently fails its own
validity check in a way that looks like noisy data rather than a wiring fault.

Since the Pi needs its own PSU regardless, USB gains nothing over Ethernet.

---

## Flashing (no monitor needed)

Raspberry Pi Imager provisions everything before first boot, so the device can go
straight to headless.

1. **OS → "Raspberry Pi OS Lite (64-bit)"**

   **64-bit is a requirement, not a preference.** On 32-bit ARM every `uint64_t`
   operation is synthesised from 32-bit pairs, so
   [E-2](../ARCHITECTURE.md#9-open-questions-and-planned-experiments) would measure
   the compiler's 64-bit emulation instead of the hardware. `scripts/run_on_pi.sh`
   refuses to run on `armv7l` for this reason.

   *Lite* — no desktop session means less background noise during benchmarks.

2. **Ctrl+Shift+X** (advanced options), before writing:
   - **Hostname:** `bincv-pi`
   - **Enable SSH → "Allow public-key authentication only"**
   - Paste the contents of `~/.ssh/id_ed25519.pub`
   - Set username
   - Locale/timezone
   - WiFi only if not using Ethernet

   Key-based auth is what lets automated sessions run without password prompts.

3. Write, insert, connect Ethernet, power on with the **official PSU**.

---

## Network

**Correction (verified 2026-08-16):** an earlier version of this file claimed
`.local` does not resolve from WSL2. That was wrong — it was concluded from a test
against a hostname that did not exist yet, which shows only that a name is absent,
not that the mechanism is broken. With a real device on the LAN, `ryan-pi4.local`
resolves fine through the Windows resolver.

The real hazard is different and worse, because it is **intermittent**: mDNS
returns **dual-stack**, and a WSL2 host typically has no IPv6 route. ssh then
sometimes picks the AAAA record and fails with `Network is unreachable`. Measured
here: it worked 10 times in a row, then failed inside the runner.

**Therefore: force IPv4.** `scripts/run_on_pi.sh` passes `-4` on every ssh and
rsync call for this reason. If you connect by hand, use `ssh -4`. A DHCP
reservation plus a literal IP in `~/.ssh/config` avoids the question entirely and
is still the more robust setup.

1. Find the Pi's address from the router's client list, or from Windows:
   ```powershell
   arp -a | Select-String "b8-27-eb|dc-a6-32|e4-5f-01"   # Raspberry Pi OUIs
   ```
2. **Set a DHCP reservation on the router** so it never moves.
3. Add to `~/.ssh/config` in WSL:
   ```
   Host bincv-pi
       HostName 10.0.0.XX          # the reserved address
       User <username>
       IdentityFile ~/.ssh/id_ed25519
   ```
4. Verify: `ssh bincv-pi uname -m` → must print `aarch64`.

### WSL2 notes

- **NAT reaches the LAN fine.** Verified on this machine: WSL pings both the router
  and the Windows host with no loss. No WSL network reconfiguration is needed.
- **A VPN can break it.** If the router pings but the Pi does not, check whether a
  VPN client is routing LAN traffic through its tunnel.
- **Mirrored networking is not recommended here.** `networkingMode=mirrored` would
  make `.local` work, but it has known friction with VPN and VMware adapters. A
  fixed IP solves the same problem with no blast radius.

### Direct PC↔Pi cable, if the router is inconvenient

Pi 4 is auto-MDIX, so a normal cable works. Assign static IPs on both ends. The Pi
still needs its own PSU. More fiddly than the LAN route — WSL2 NAT routing to a
link-local Windows adapter is awkward — so prefer the router unless there is a
reason not to.

---

## First-connection checklist

```bash
ssh bincv-pi 'uname -m'                                  # must be aarch64
ssh bincv-pi 'vcgencmd get_throttled'                    # want 0x0
ssh bincv-pi 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
ssh bincv-pi 'sudo apt-get update && sudo apt-get install -y build-essential cmake rsync'
```

`vcgencmd get_throttled` returns a bitmask. `0x0` is the only good answer — any
non-zero value means undervoltage or thermal throttling has occurred, and
measurements taken in that state are invalid rather than merely slow.

Then use the runner (see [T1.10](../TASKS.md)):

```bash
./scripts/run_on_pi.sh bincv-pi ./tests/test_binMat     # expect 261/261
```

It enforces the measurement discipline — architecture check, throttle checks
before and after, `performance` governor with restore-on-exit, core pinning — so
those constraints do not depend on being remembered.

---

## What works without it

The Pi closes experiments; it is not needed to write the library.

**Fully unblocked:** all of Phase 1 (T1.1–T1.9) and the Phase 2 implementation
tasks (T2.1–T2.7). These are correctness and structural work.

**Measurable without it:** *every footprint result.* Allocated bytes are
architecture-independent, so half of binCV's two co-equal goals can be developed
and measured on any machine. [X-1](../EXPERIMENTS.md)'s footprint half already
closed on x86.

**Visible without it:** algorithmic wins. Bit-parallel versus per-pixel is a
10×-class difference that shows anywhere. The Pi is for *micro-decisions* —
alignment padding, word width, whether an accumulator stays L1-resident — not for
validating the core approach.

**Blocked:** T2.8, T2.9, T2.10 (closing E-1, E-2, E-3), Phase 4 validation, and
meaningful Phase 5 NEON work.

**Why the blocked set is small:** two architecture decisions insulate the
structural work from the tuning results.
[D-5](../ARCHITECTURE.md#d-5-views-are-core-not-an-add-on) gives kernels views with
*runtime* stride, so changing the alignment default never touches a kernel.
[D-1](../ARCHITECTURE.md#d-1-template-on-the-word-type-not-a-bit-count) templates
on the word type, so changing the default word width never touches a kernel either.
Both experiments can therefore land late without invalidating work already done.
