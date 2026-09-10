#!/usr/bin/env python3
"""Gate every [REF:key#qN] and [MEAS:key] citation against something on disk.

    python dev-notes/whoop-rs/check_evidence.py

Exit 0 = every citation in every scanned document resolves to a retrieved artifact whose text
actually contains the quoted span. Exit 1 = at least one does not.

What it does NOT do: judge whether a quote SUPPORTS the claim. A human does that. What it makes
impossible is attributing words to a paper that does not contain them, which is the specific way
this project has published wrong claims three times.

Run it before any document that cites evidence is treated as backing.
"""
from __future__ import annotations

import hashlib
import html
import re
import sys
from pathlib import Path

# Quotes carry arrows, Greek and dashes; the Windows console is cp1252 and a failure report
# printing one crashes the gate instead of reporting it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "whoop-research" / "_actual" / "papers"
MEAS = ROOT / "whoop-research" / "_actual" / "meas"

# Documents whose citations are gated. Add new ones here; a document not listed is not gated,
# and therefore is not evidence.
SCANNED = [
    ROOT / "dev-notes" / "TANV1-TARGET-AUDIT-20260901.md",
    ROOT / "dev-notes" / "whoop-rs" / "HLD-LLD-SLEEP-MODEL.md",
    ROOT / "dev-notes" / "whoop-rs" / "SPEC-EMISSION-LADDER.md",
    ROOT / "whoop-research" / "_actual" / "CLAIMS.md",
    ROOT / "whoop-research" / "_actual" / "notes" / "synthesis" / "FEATURE-ROSTER.md",
    ROOT / "whoop-research" / "_actual" / "notes" / "synthesis" / "KAPPA-ALTERNATIVES.md",
]

REF_RE = re.compile(r"\[REF:([a-z0-9]+)#(q\d+)\]")
MEAS_RE = re.compile(r"\[MEAS:([a-z0-9_-]+)\]")
QUOTE_RE = re.compile(r"^##\s+(q\d+)\s*$", re.M)

MEAS_REQUIRED = ["what:", "data:", "harness:", "date:", "result:", "falsified_by:"]


def is_corrupt(quote: str) -> bool:
    """A quote carrying U+FFFD was transcribed from a broken extraction.

    `normalise` folds every non-ASCII character to a space, so `���min` and the correct
    `\U0001d451min` both become " min" and MATCH. Two quotes reached the ledger that way and passed
    every run of the gate. Verbatimness cannot catch this; only refusing the character can.
    """
    return "�" in quote


def normalise(text: str, markup: bool = False) -> str:
    """Fold the things that differ between a paste and an artifact.

    Deliberately conservative: entities, a few typographic characters, whitespace and case. It does
    NOT stem, reorder or drop punctuation, so a paraphrase still fails.

    `markup=True` additionally strips tags, and it must ONLY be set for genuine HTML/XML artifacts.
    The tag pattern `<[^>]+>` matches newlines, so on PLAIN TEXT a `<` from one inequality and a `>`
    from a later one delete everything between them. Measured on our own corpus: **35% of
    rossi2025's text and 43% of schuetz2026's** were invisible to the gate, so a genuinely verbatim
    quote from those regions failed with "Either it is a paraphrase or it came from memory" -- a
    FALSE NEGATIVE indistinguishable from the real thing. Found by two synthesis agents
    independently, not by the self-test, because every mutation happened to sit outside a swallowed
    region.
    """
    if markup:
        text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # Rejoin a word split by a hyphen across whitespace. One extractor de-hyphenates and the other
    # keeps "re-\nporting", so without this the SAME sentence in the SAME paper matches under one
    # tool and not the other - a difference with nothing to do with what the paper says. Matching
    # ANY whitespace, not just a newline, is deliberate: a quote pasted from one rendering carries
    # "Bayes- optimal" where the artifact has "Bayes-\noptimal", and only folding both to the same
    # string makes them agree. Runs before the whitespace collapse, which would hide the newline.
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    for a, b in (("±", "+/-"), ("¼", "1/4"), ("¾", "3/4"),
                 ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    text = re.sub(r"[^\x20-\x7e]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def strip_code(text: str) -> str:
    """Drop fenced blocks and inline code spans.

    A citation inside backticks is being SHOWN, not made -- the contract docs and this ledger both
    write `[MEAS:key]` as an example of the syntax. Scanning those would demand a paper named "key".
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", " ", text)


def parse_meta(path: Path) -> dict[str, str]:
    """Flat key: value scrape. Not a YAML parser -- it only needs the leaf fields we gate on."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # [a-z_0-9]: the key `sha256` has digits, and excluding them silently skipped the artifact
        # hash check entirely -- caught by the self-test's tamper mutation, not by reading the code.
        m = re.match(r"^\s*([a-z_0-9]+):\s*(.*)$", line)
        if m and m.group(2).strip():
            out.setdefault(m.group(1), m.group(2).strip().strip('"'))
    return out


def load_quotes(path: Path) -> dict[str, str]:
    """Split quotes.md into {qN: quoted text}. The quote is the blockquote under the heading."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    marks = list(QUOTE_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        quoted = " ".join(ln.lstrip("> ").rstrip() for ln in body.splitlines() if ln.startswith(">"))
        out[m.group(1)] = quoted.strip()
    return out


def check_ref(key: str, qid: str, fails: list[str]) -> None:
    d = REFS / key
    if not d.is_dir():
        fails.append(f"[REF:{key}#{qid}] -- papers/{key}/ does not exist")
        return
    meta_p, quotes_p = d / "meta.yaml", d / "quotes.md"
    if not meta_p.exists():
        fails.append(f"[REF:{key}#{qid}] -- no meta.yaml")
        return
    if not quotes_p.exists():
        fails.append(f"[REF:{key}#{qid}] -- no quotes.md")
        return

    meta = parse_meta(meta_p)
    art = d / meta.get("file", "")
    if not meta.get("file") or not art.exists():
        fails.append(f"[REF:{key}#{qid}] -- artifact '{meta.get('file')}' missing; a reference "
                     f"without the retrieved primary is a recollection")
        return

    raw = art.read_bytes()
    want = meta.get("sha256", "")
    got = hashlib.sha256(raw).hexdigest()
    if not want:
        fails.append(f"[REF:{key}#{qid}] -- meta.yaml has no sha256; the artifact is unpinned, so "
                     f"the quotes are not checked against a known file")
        return
    if want != got:
        fails.append(f"[REF:{key}#{qid}] -- artifact sha256 changed ({want[:12]}.. -> {got[:12]}..); "
                     f"the quotes were checked against a different file")
        return

    quotes = load_quotes(quotes_p)
    if qid not in quotes:
        have = ", ".join(sorted(quotes)) or "none"
        fails.append(f"[REF:{key}#{qid}] -- no such quote (have: {have})")
        return
    if not quotes[qid]:
        fails.append(f"[REF:{key}#{qid}] -- quote is empty")
        return
    if is_corrupt(quotes[qid]):
        fails.append(f"[REF:{key}#{qid}] -- the quote carries U+FFFD, so it was taken from a broken "
                     f"extraction; re-transcribe it from the artifact")
        return

    # A PDF's body text lives in compressed streams, so decoding the raw bytes as UTF-8 finds
    # NOTHING except the front matter. Measured: 0 of 5 real body phrases from beattie2017 were
    # found that way, while the title -- which sits uncompressed in the metadata -- was. That made
    # the first version of this check look like it worked on PDFs when it did not, and it would have
    # rejected every CORRECT quote from 6 of our 8 papers as a paraphrase.
    #
    # So a PDF artifact must carry `text.txt`, extracted at ingest by `pdftotext` and pinned by its
    # own sha256. Quotes are checked against that.
    text_p = d / "text.txt"
    if art.suffix.lower() == ".pdf":
        if not text_p.exists():
            fails.append(f"[REF:{key}#{qid}] -- artifact is a PDF but there is no text.txt; run "
                         f"`extract_pdf_text.py`. Quotes cannot be checked against PDF bytes")
            return
        want_t = meta.get("text_sha256", "")
        got_t = hashlib.sha256(text_p.read_bytes()).hexdigest()
        if not want_t:
            fails.append(f"[REF:{key}#{qid}] -- meta.yaml has no text_sha256; text.txt is unpinned")
            return
        if want_t != got_t:
            fails.append(f"[REF:{key}#{qid}] -- text.txt sha256 changed "
                         f"({want_t[:12]}.. -> {got_t[:12]}..); the quotes were checked against "
                         f"a different extraction")
            return
        source = text_p.read_bytes()
    else:
        source = raw

    # Strip tags only for real markup. `text.txt` is pdftotext output and `.md` is prose; both can
    # contain bare `<` and `>` from inequalities, and stripping there destroys content.
    markup = art.suffix.lower() in (".html", ".htm", ".xml") and source is raw
    body = normalise(source.decode("utf-8", errors="replace"), markup=markup)
    if normalise(quotes[qid], markup=markup) not in body:
        fails.append(f"[REF:{key}#{qid}] -- QUOTE NOT FOUND IN THE ARTIFACT. "
                     f"Either it is a paraphrase or it came from memory:\n"
                     f"        {quotes[qid][:110]}")


def check_meas(key: str, fails: list[str]) -> None:
    p = MEAS / f"{key}.md"
    if not p.exists():
        fails.append(f"[MEAS:{key}] -- meas/{key}.md does not exist; a number with no measurement "
                     f"record is a recollection")
        return
    text = p.read_text(encoding="utf-8")
    # Anchored at line start. A plain `in` test passes for `was_falsified_by:`, which the self-test
    # caught -- the substring was present while the field was not.
    missing = [f for f in MEAS_REQUIRED
               if not re.search(rf"^{re.escape(f)}", text, re.M)]
    if missing:
        fails.append(f"[MEAS:{key}] -- missing required field(s): {', '.join(missing)}")


SELF_TESTS = [
    ("a paraphrase replacing a verbatim quote",
     "whoop-research/_actual/papers/bizzotto2018/quotes.md",
     "stages 3 and 4 were merged together and movement time epochs were removed",
     "stages three and four were combined and movement epochs discarded"),
    ("a citation to a paper never retrieved",
     "whoop-research/_actual/CLAIMS.md", "[REF:varga2016#q1]", "[REF:yuanlin2006#q1]"),
    ("a measurement record with no falsifier",
     "whoop-research/_actual/meas/deep-dwell-tail.md", "falsified_by:", "was_falsified_by:"),
    ("a quote id that does not exist",
     "whoop-research/_actual/CLAIMS.md", "[REF:bizzotto2018#q2]", "[REF:bizzotto2018#q99]"),
    ("the artifact edited under quotes already checked against it",
     "whoop-research/_actual/papers/varga2016/artifact.html", "<html", "<!-- tampered --><html"),
    ("a PDF's extracted text edited under quotes already checked against it",
     "whoop-research/_actual/papers/fonseca2018/text.txt",
     "all features were normalized to have zero mean",
     "all features were left unnormalized"),
    # Regression guard for the greedy tag-strip FALSE NEGATIVE. Wrapping a quoted span in bare
    # angle brackets used to delete it from the gate's view, so the quote failed as a "paraphrase".
    # With markup stripping off for plain text the span survives and the quote still matches, so
    # this mutation must be caught by the CONTENT change, not by the brackets.
    ("plain-text angle brackets swallowing a quoted span",
     "whoop-research/_actual/papers/fonseca2018/text.txt",
     "to reduce between-subject physiological and equipment-related variations",
     "to reduce between-subject physiological and <equipment-related> variations"),
]


def self_test() -> int:
    """Break the gate five ways and require it to catch each.

    A gate nobody has broken proves nothing. This one shipped green while checking ZERO citations,
    because an over-eager code-span strip had eaten every real one. Only the mutations found it.
    """
    print(f"self-test: mutating {len(SELF_TESTS)} things that MUST fail the gate\n")
    bad = 0
    for name, rel, old, new in SELF_TESTS:
        p = ROOT / rel
        original = p.read_bytes()
        try:
            # Operate on BYTES. Decoding strictly crashed here the moment a pdftotext extraction
            # arrived in the locale encoding rather than UTF-8 -- the mutation harness must not
            # assume anything about an artifact's encoding.
            if old.encode() not in original:
                print(f"  INCONCLUSIVE  {name}: anchor not present in {rel}")
                bad += 1
                continue
            p.write_bytes(original.replace(old.encode(), new.encode(), 1))
            caught = run_checks(quiet=True) != 0
        finally:
            p.write_bytes(original)
        print(f"  {'caught  ' if caught else 'MISSED  '}{name}")
        bad += 0 if caught else 1

    if run_checks(quiet=True) != 0:
        print("\n  MISSED  the tree did not return to green after restore")
        bad += 1
    print("\nself-test " + ("OK" if not bad else f"FAILED -- {bad} mutation(s) not caught"))
    return 1 if bad else 0


def run_checks(quiet: bool = False) -> int:
    fails: list[str] = []
    refs_seen: set[tuple[str, str]] = set()
    meas_seen: set[str] = set()
    per_doc: list[tuple[str, int, int]] = []

    for doc in SCANNED:
        if not doc.exists():
            if not quiet:
                print(f"  skip (absent): {doc.relative_to(ROOT)}")
            continue
        text = strip_code(doc.read_text(encoding="utf-8"))
        r = REF_RE.findall(text)
        m = MEAS_RE.findall(text)
        per_doc.append((doc.name, len(r), len(m)))
        for key, qid in r:
            refs_seen.add((key, qid))
        for key in m:
            meas_seen.add(key)

    for key, qid in sorted(refs_seen):
        check_ref(key, qid, fails)
    for key in sorted(meas_seen):
        check_meas(key, fails)

    if not quiet:
        for name, nr, nm in per_doc:
            # Zero is printed loudly. A green gate over zero citations is the failure mode this
            # script itself had on its first run.
            flag = "   <-- NO CITATIONS" if nr + nm == 0 else ""
            print(f"  {name:<32} {nr:>3} ref  {nm:>3} meas{flag}")
        print(f"total: {len(refs_seen)} reference citation(s), {len(meas_seen)} measurement "
              f"citation(s) across {len(per_doc)} document(s)")
        orphan = sorted(d.name for d in REFS.iterdir()
                        if d.is_dir() and d.name not in {k for k, _ in refs_seen}) \
            if REFS.is_dir() else []
        if orphan:
            print(f"note: refs on disk not yet cited: {', '.join(orphan)}")

    if fails:
        if not quiet:
            print(f"\nFAIL -- {len(fails)} citation(s) do not resolve:\n")
            for f in fails:
                print(f"  - {f}")
        return 1
    if not quiet:
        print("OK -- every citation resolves to a retrieved artifact containing the quoted span")
    return 0


def sweep_quotes(quiet: bool = False) -> int:
    """Check EVERY quote in every `quotes.md`, cited or not.

    `run_checks` only reaches a quote once something cites it, so a wrong quote can sit on disk
    indefinitely and fail the day it is first used. This sweeps the lot.
    """
    total, fails = 0, []
    for d in sorted(p for p in REFS.iterdir() if p.is_dir()):
        q = d / "quotes.md"
        if not q.exists():
            continue
        text = q.read_text(encoding="utf-8")
        if "None recorded yet" in text:
            continue
        art = d / "text.txt"
        if not art.exists():
            art = next((f for f in sorted(d.iterdir()) if f.suffix in (".xml", ".html", ".htm")), None)
        if art is None:
            continue
        hay = normalise(art.read_text(encoding="utf-8", errors="replace"),
                        markup=art.suffix in (".xml", ".html", ".htm"))
        for m in re.finditer(r"^##\s+(q\d+)\s*$(.*?)(?=^##\s+q|\Z)", text, re.M | re.S):
            span = " ".join(ln[1:].strip() for ln in m.group(2).splitlines() if ln.startswith(">"))
            if not span.strip():
                continue
            total += 1
            if is_corrupt(span):
                fails.append(f"{d.name}#{m.group(1)}: CORRUPT - carries U+FFFD, re-transcribe from "
                             f"the artifact: {span[:70]}")
            elif normalise(span) not in hay:
                fails.append(f"{d.name}#{m.group(1)}: {span[:90]}")
    if not quiet:
        if fails:
            print(f"FAIL -- {len(fails)} of {total} quote(s) on disk are not verbatim:\n")
            for f in fails:
                print(f"  - {f}")
        else:
            print(f"OK -- all {total} quote(s) on disk are verbatim, cited or not")
    return 1 if fails else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    if "--all-quotes" in sys.argv:
        return sweep_quotes()
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
