#!/usr/bin/env python3
"""Generate docs/API.md from the public headers.

WHY THIS IS GENERATED AND NOT WRITTEN. Every public entry point in binCV already
carries a `@brief` and states its API tier -- that was a rule before there was any
reference to put them in.
So the reference is a VIEW of the headers, and a hand-written one would drift from them
the first time a signature moved.

Run: python3 scripts/gen_api_index.py [--out PATH]

Entries inside an `impl` or `detail` namespace are internal by convention and are
skipped, like anything marked INTERNAL. The number of public entries whose brief
states no API tier is printed to stderr, because the tier is part of the contract.
"""
import argparse
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INC = ROOT / "include" / "bincv"
OUT = ROOT / "docs" / "API.md"

# A declaration we consider public API: a function, type, enum or named constant
# at namespace scope. Constants ride on the k-prefix convention, which is what
# keeps documented struct MEMBERS with initializers out of the index.
DECL = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?"
    r"(?:inline\s+|constexpr\s+|static\s+)*"
    r"(?:(?P<kind>struct|class|enum\s+class)\s+(?P<type>\w+)"
    r"|[\w:<>,&*\s]+?\b(?P<var>k[A-Z]\w*)\s*(?:\[[^\]]*\])?\s*="
    r"|[\w:<>,&*\s]+?\b(?P<fn>\w+)\s*\()"
)
TIER = re.compile(r"\*\*API TIER (\d)|\*\*INTERNAL", re.I)
# The internal namespaces inside a public header. They close with the `} // namespace
# impl` comment throughout, which is what makes tracking them a line match rather
# than a brace count.
NS_OPEN = re.compile(r"^\s*namespace\s+(impl|detail)\s*\{")
NS_CLOSE = re.compile(r"^\s*\}\s*//\s*namespace\s+(impl|detail)\b")
# GCC's function attributes sit where a signature starts and are not a name.
ATTRIBUTE = re.compile(r"__attribute__\s*\(\(.*?\)\)\s*")


def first_sentence(text):
    """The brief's first sentence. Docstrings here run to paragraphs of rationale --
    valuable in the header, useless in a table."""
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"^(.*?[.!?])(?:\s|$)", text)
    s = (m.group(1) if m else text).strip()
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    return s.rstrip(".")


def briefs(path):
    """Yield (name, kind, brief, tier) for each documented public declaration."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    out, block, seen = [], [], set()
    hidden = 0                            # depth inside an impl/detail namespace
    for i, raw in enumerate(lines):
        if NS_OPEN.match(raw):
            hidden += 1
        elif NS_CLOSE.match(raw) and hidden > 0:
            hidden -= 1
        s = raw.strip()
        if s.startswith("///"):
            block.append(s[3:].strip())
            continue
        if not block:
            continue
        if not s or s.startswith("//"):
            block = []
            continue
        text = " ".join(block)
        block = []
        m = re.search(r"@brief\s+(.*?)(?:\s*@\w|$)", text, re.S)
        if not m:
            continue
        tm = TIER.search(text)
        if tm and not tm.group(1):
            continue                      # said INTERNAL; not public API
        tier = tm.group(1) if tm else ""
        if hidden:
            continue                      # inside namespace impl/detail; not public API
        # A template declaration puts `template <...>` on its own line and the signature
        # on the next, so join forward -- otherwise every templated entry point is lost.
        decl = ATTRIBUTE.sub("", s)
        d, k = DECL.match(decl), 0
        while d is None and k < 3 and i + k + 1 < len(lines):
            k += 1
            decl = decl + " " + ATTRIBUTE.sub("", lines[i + k].strip())
            d = DECL.match(decl)
        if d is None:
            continue
        name = d.group("type") or d.group("var") or d.group("fn")
        if (not name or name.startswith("operator") or name.startswith("__")
                or name in ("if", "for", "return")):
            continue
        kind = (d.group("kind") or ("constant" if d.group("var") else "function"))
        kind = kind.replace("enum class", "enum")
        key = (name, kind)
        if key in seen:
            continue                      # overloads collapse to one row
        seen.add(key)
        out.append((name, kind, first_sentence(m.group(1)), tier))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT),
                    help="where to write the reference (default: docs/API.md)")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)
    groups = []
    for sub in ("", "ops", "io", "core", "threads"):
        d = INC / sub if sub else INC
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.hpp")):
            entries = briefs(path)
            if entries:
                rel = path.relative_to(ROOT)
                groups.append((f"{sub + '/' if sub else ''}{path.name}", rel, entries))

    lines = [
        "# binCV API reference",
        "",
        "**Generated** by `scripts/gen_api_index.py` from the headers — do not edit.",
        "Every entry is the `@brief` from the declaration itself, so this cannot drift",
        "from the code without the code changing.",
        "",
        "## API tiers",
        "",
        "| tier | meaning |",
        "|---|---|",
        "| **1** | **bit-exact against OpenCV**, proven by a test |",
        "| **2** | same role and call shape as an OpenCV function, different numerics |",
        "| **3** | no OpenCV equivalent; deliberately does not borrow an OpenCV name |",
        "",
        "Anything marked INTERNAL in its docstring is omitted here.",
        "",
        "## Contents",
        "",
    ]
    for name, _, entries in groups:
        anchor = name.replace("/", "").replace(".", "")
        lines.append(f"- [`{name}`](#{anchor}) — {len(entries)} entries")
    lines.append("")
    for name, rel, entries in groups:
        lines.append(f"## `{name}`")
        lines.append("")
        lines.append(f"[`{rel}`]({'../' + str(rel)})")
        lines.append("")
        lines.append("| | tier | |")
        lines.append("|---|---|---|")
        for n, kind, brief, tier in entries:
            label = f"`{n}`" if kind == "function" else f"`{n}` *({kind})*"
            lines.append(f"| {label} | {tier or '—'} | {brief} |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(e) for _, _, e in groups)
    print(f"{args.out}: {len(groups)} headers, {total} entries")
    untiered = [(name, n) for name, _, entries in groups for n, _, _, tier in entries if not tier]
    if untiered:
        print(f"{len(untiered)} public entries state no API tier:", file=sys.stderr)
        for header, n in untiered:
            print(f"  {header}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
