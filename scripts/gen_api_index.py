#!/usr/bin/env python3
"""Generate docs/API.md from the public headers.

WHY THIS IS GENERATED AND NOT WRITTEN. Every public entry point in binCV already
carries a `@brief` and states its API tier -- that was a rule before there was any
reference to put them in (CLAUDE.md: "State the API tier in every public docstring").
So the reference is a VIEW of the headers, and a hand-written one would drift from them
the first time a signature moved.

Run: python3 scripts/gen_api_index.py
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
INC = ROOT / "bincv-cpp" / "include" / "bincv-cpp"
OUT = ROOT / "docs" / "API.md"

# A declaration we consider public API: a function, type or enum at namespace scope.
DECL = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?"
    r"(?:inline\s+|constexpr\s+|static\s+)*"
    r"(?:(?P<kind>struct|class|enum\s+class)\s+(?P<type>\w+)"
    r"|[\w:<>,&*\s]+?\b(?P<fn>\w+)\s*\()"
)
TIER = re.compile(r"\*\*API TIER (\d)|\*\*INTERNAL", re.I)


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
    for i, raw in enumerate(lines):
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
        # A template declaration puts `template <...>` on its own line and the signature
        # on the next, so join forward -- otherwise every templated entry point is lost.
        decl, d, k = s, DECL.match(s), 0
        while d is None and k < 3 and i + k + 1 < len(lines):
            k += 1
            decl = decl + " " + lines[i + k].strip()
            d = DECL.match(decl)
        if d is None:
            continue
        name = d.group("type") or d.group("fn")
        if not name or name.startswith("operator") or name in ("if", "for", "return"):
            continue
        kind = (d.group("kind") or "function").replace("enum class", "enum")
        key = (name, kind)
        if key in seen:
            continue                      # overloads collapse to one row
        seen.add(key)
        out.append((name, kind, first_sentence(m.group(1)), tier))
    return out


def main():
    groups = []
    for sub in ("", "ops", "io", "core", "threads"):
        d = INC / sub if sub else INC
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.hpp")):
            entries = briefs(path)
            if entries:
                rel = path.relative_to(INC.parent.parent.parent)
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
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(e) for _, _, e in groups)
    print(f"docs/API.md: {len(groups)} headers, {total} entries")


if __name__ == "__main__":
    main()
