#!/usr/bin/env python3
"""Block-level diff of two border-card captures.

    python whoop-rs/tools/card_diff.py OLD.txt NEW.txt
    python whoop-rs/tools/card_diff.py --self-test

`diff` on two captures is unreadable: the card prints 61 blocks of 20-odd lines and a one-number
change anywhere shifts nothing, so every real move is buried in a wall of identical context. Worse,
the capture opens with cargo's build timing, which differs on every run and makes two identical
cards look changed.

So this splits on the `== <cohort>  arm: <arm>  n=<n> ==` headers and compares block against block.
Anything before the first header -- the compile line, the build timing, the banner -- is never
compared. The header's `n=` is folded into the block's first line rather than its key, so an arm
that scored a different number of recordings is a line that moved and not a block that vanished.

What it reports: blocks only in A, blocks only in B, and for a shared block the exact lines that
differ, old above new. Stdlib only.
"""

import difflib
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^== (?P<cohort>\S+)\s+arm: (?P<arm>.*?)\s+n=(?P<n>\d+) ==\s*$")


def blocks(text):
    """Split a capture into `{cohort / arm: [lines]}`. A line before the first header belongs to no
    block and is dropped, which is what keeps the build timing out of the comparison."""
    out, key, buf = {}, None, []
    for line in text.splitlines():
        m = HEADER.match(line.rstrip())
        if m:
            if key is not None:
                out[key] = buf
            key = f"{m['cohort']} / {m['arm']}"
            buf = [f"n={m['n']}"]
        elif key is not None:
            buf.append(line.rstrip())
    if key is not None:
        out[key] = buf
    return out


def block_diff(old, new):
    """The lines that differ inside one block, as `('-', line)` and `('+', line)` in reading order.

    `difflib` rather than a positional zip: a block that gained or lost a line, and a card does
    when a Bland-Altman row turns SLOPED, would otherwise report every line after it as changed.
    """
    return [(d[0], d[2:]) for d in difflib.ndiff(old, new) if d[0] in "-+"]


def report(a_text, b_text, out=print):
    """Print the block-level differences and return how many shared blocks moved."""
    a, b = blocks(a_text), blocks(b_text)
    gone, added = [k for k in a if k not in b], [k for k in b if k not in a]
    shared = [k for k in b if k in a]
    for label, keys in (("only in A", gone), ("only in B", added)):
        out(f"{label}: {len(keys)} block(s)")
        for k in keys:
            out(f"  == {k}")
    moved = 0
    for k in shared:
        d = block_diff(a[k], b[k])
        if not d:
            continue
        moved += 1
        out(f"\n== {k}")
        for sign, line in d:
            out(f"  {sign} {line}")
    out(f"\n{moved} of {len(shared)} shared block(s) differ")
    return moved


CARD_A = """   Finished `release` profile in 1.77s
THE BORDER
== dreamt  arm: the null  n=100 ==
  macro F1 0.4443  min recall 0.1723
  rem        0.388
aauwss: 13 recording(s)
== aauwss  arm: gone in B  n=13 ==
  macro F1 0.1000
"""

CARD_B = """   Finished `release` profile in 41.02s
THE BORDER
== dreamt  arm: the null  n=99 ==
  macro F1 0.4443  min recall 0.1723
  rem        0.388
  a line that only B has
aauwss: 13 recording(s)
== aauwss  arm: new in B  n=13 ==
  macro F1 0.2000
"""


def self_test():
    """The claims: the build timing is invisible, `n=` is a line and not an identity, a block that
    gained a line does not report every line after it as changed, and the cohort header between two
    blocks is real content that rides with the block above it rather than being thrown away."""
    a, b = blocks(CARD_A), blocks(CARD_B)
    assert list(a) == ["dreamt / the null", "aauwss / gone in B"], list(a)
    assert "Finished" not in "".join(a["dreamt / the null"]), "the build timing must not be compared"
    assert "aauwss: 13 recording(s)" in a["dreamt / the null"], a["dreamt / the null"]

    lines = []
    moved = report(CARD_A, CARD_B, out=lines.append)
    text = "\n".join(lines)
    assert moved == 1, moved
    assert "== aauwss / gone in B" in text and "== aauwss / new in B" in text
    assert "only in A: 1 block(s)" in text and "only in B: 1 block(s)" in text

    d = block_diff(a["dreamt / the null"], b["dreamt / the null"])
    assert ("-", "n=100") in d and ("+", "n=99") in d, d
    assert ("+", "  a line that only B has") in d, d
    # The unchanged lines sit BETWEEN those two changes. A positional zip would have called all of
    # them changed as well, and that is the whole reason difflib is in here.
    assert len(d) == 3, d
    assert not any("0.4443" in line for _, line in d), d

    # Two identical captures differ nowhere, even though their timings do not match.
    assert report(CARD_A, CARD_A.replace("1.77s", "9.99s"), out=lambda _: None) == 0
    print("card_diff self-test OK")
    return 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    report(Path(args[0]).read_text(encoding="utf-8", errors="replace"),
           Path(args[1]).read_text(encoding="utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
