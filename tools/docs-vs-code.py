#!/usr/bin/env python3
"""Re-derive every countable claim in `whoop-rs/docs/` from the source, and fail when the two disagree.

    python whoop-rs/tools/docs-vs-code.py            # derive everything, including the two cargo runs
    python whoop-rs/tools/docs-vs-code.py --no-tests  # skip the cargo runs, check the static claims only

`az-sweep.py` pins the A-Z manifest against what the harnesses PRINT and `manifest-vs-docs.py` pins the
manifest against the dev-notes documents. Both edges of that triangle end at the manifest, and the manifest
names only the two dev-notes files - so the SHIPPED docs under `whoop-rs/docs/` were pinned by nothing.
They drifted: a test count stale by 137, an `#[ignore]` count off by one, an FFI export count off by 22,
one staging kappa written two ways in one document, and four algorithms tagged unwired that Kotlin calls.

A count can also drift while this gate reads green, if the gate asks a question loose enough to be
answered by something else. Both "called from Kotlin" checks did: a bare name search found `hrZonesForAge`
in `RustScores`'s own wrapper declaration whether or not it still reached Rust, and read `client_hello` as
called because an unrelated `DeviceFamily.clientHello` byte array exists — which is where the published
"19 of 31 codec methods" came from. Both now name the receiver, so only a real crossing counts.

A free function is qualified by `uniffi.whoop_ffi.<name>` and a namesake cannot reach it. An object method
has no such namespace, and four matchers in a row published a wrong integer for one — 19, then 18, then 8.
So that one claim is a NAMED LIST in the document rather than a total: the extraction is still approximate,
but its errors are now legible, and a receiver it cannot resolve is reported instead of counted.

Every claim below names the document, the regex that extracts what the document says, and the command or
source scan that says what is true. A claim whose regex no longer matches is a FAILURE, not a skip: a
sentence rewritten out of the document must be re-pinned deliberately, or this gate quietly stops covering
it. Exit status is 1 when anything disagrees.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "docs"
WHOOP = ROOT.parent
KOTLIN = WHOOP / "noop-wt-tan" / "android" / "app" / "src" / "main" / "java"
FIXTURES = WHOOP / "sleep-benchmark" / "fixtures_multi_clean3"

# Small counts a document spells out rather than digits. Comparison is case-insensitive, so the word in
# the prose is pinned as the word, not silently accepted because a digit happened to match.
SPELLED = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def rs_files(rel: str) -> list[Path]:
    return sorted((ROOT / rel).rglob("*.rs"))


def dataset_gate_attrs() -> int:
    """`#[ignore]` ATTRIBUTES on the sleep-dataset parity gates, which is the population the sentence
    in `algorithms.md` describes.

    This replaced a workspace-wide scan for the literal string `#[ignore` that answered 62 while the
    document said four. Three different populations were in play and none of them was the sentence's:
    the scan counted every crate's ecg and sensitivity harnesses as well, and counted the string where
    it appears inside a comment (`dataset_parity.rs` mentions it twice in prose). A gate whose
    population is wider than the claim can only ever be red.
    """
    t = (ROOT / "crates" / "physio-algo" / "tests" / "dataset_parity.rs").read_text(encoding="utf-8")
    return len(re.findall(r"^\s*#\[ignore", t, re.M))


def ffi_exports() -> list[str]:
    """Free functions on the uniffi surface (the `WhoopCodec` object's methods are counted separately)."""
    out = []
    for p in sorted((ROOT / "crates" / "whoop-ffi" / "src").glob("*.rs")):
        t = p.read_text(encoding="utf-8")
        out += re.findall(r"#\[uniffi::export\][^\n]*\n(?:\s*//[^\n]*\n)*\s*pub fn (\w+)", t)
    return sorted(out)


def codec_methods() -> list[str]:
    t = (ROOT / "crates" / "whoop-ffi" / "src" / "codec.rs").read_text(encoding="utf-8")
    return re.findall(r"\n    pub fn (\w+)", t[t.index("impl WhoopCodec {"):])


def camel(s: str) -> str:
    head, *tail = s.split("_")
    return head + "".join(w.capitalize() for w in tail)


def kotlin_files() -> list[str]:
    if not KOTLIN.is_dir():
        return []
    return [p.read_text(encoding="utf-8", errors="replace") for p in KOTLIN.rglob("*.kt") if p.name != "whoop_ffi.kt"]


def kotlin_blob() -> str:
    return "\n".join(kotlin_files())


def free_fns_called(names: list[str]) -> int:
    """A free function is called only through a qualified `uniffi.whoop_ffi.<name>`.

    A bare-name search cannot show this: `RustScores` wraps most exports in a Kotlin function of the
    same name, so the wrapper's own declaration satisfied the search whether or not it still reached
    Rust. Deleting a call site had to remain visible here, so the receiver is part of the pattern.
    """
    reached = set(re.findall(r"uniffi\.whoop_ffi\.(\w+)", kotlin_blob()))
    return sum(1 for n in names if camel(n) in reached)


CODEC_BOUND = re.compile(r"\bva[lr]\s+(\w+)[^\n=]*(?:=|by\s+lazy\s*\{)[^\n]*\bWhoopCodec\(")
# `fun f(a: A): T = <expr>` / `val f get() = <expr>`, expression body on the declaration line.
CODEC_CHOICE = re.compile(r"^[ \t]*(?:\w+[ \t]+)*(?:fun|val)[ \t]+(\w+)[ \t]*(?:\([^)\n]*\))?[^=\n]*=[ \t]*(\S[^\n]*)$", re.M)
CHOICE_WORDS = frozenset({"if", "else", "when", "this", "it", "return"})
CHOICE_COND = re.compile(r"\b(?:if|when)[ \t]*\([^()\n]*\)")
# `recv.method(` or `accessor(args).method(`, the two receiver shapes Kotlin writes on one line.
CODEC_CALL = re.compile(r"\b([A-Za-z_]\w*)[ \t]*(?:\([^()\n]*\))?[ \t]*\.[ \t]*([A-Za-z_]\w*)[ \t]*\(")
# A declared return type, which is the ONE thing a block body cannot omit: `fun f(a: A): T {` and
# `val p: T get() {`. It stops at `=` so an initialiser is never read as a type, and ends at `=` OR the
# line, so an annotated expression body whose value is on the NEXT line is still read.
CODEC_TYPED = [
    re.compile(r"^[ \t]*(?:\w+[ \t]+)*fun[ \t]+(\w+)[ \t]*\([^\n]*?\)[ \t]*:[ \t]*([^\n=]*)(?:=|$)", re.M),
    re.compile(r"^[ \t]*(?:\w+[ \t]+)*va[lr][ \t]+(\w+)[ \t]*:[ \t]*([^\n=]*)(?:=|$)", re.M),
]


def strip_comments(src: str) -> str:
    return "\n".join(line.split("//")[0] for line in re.sub(r"/\*[\s\S]*?\*/", "", src).splitlines())


def codec_receivers(src: str) -> set[str]:
    """The identifiers in one Kotlin file known to hold a `WhoopCodec`, closed to a fixpoint.

    Two rules, both about types rather than about one file's spelling. A `val`/`var` initialised from
    `WhoopCodec(` holds one. An expression-bodied declaration holds one when every identifier left in a
    RESULT position is already bound - so the conditions of `if`/`when` are deleted first, being
    irrelevant to what the expression evaluates to, and `= if (gen == Gen.GEN5) gen5 else gen4` reduces
    to a choice between two codecs. One operation in a result position (`gen5.buzzFrame(s)`,
    `gen5.toString()`) leaves it unbound, because the result is then not a codec.

    Deleting conditions rather than allowing the parameters is what makes this survive the caller's
    spelling: the same selector rewritten from a `Boolean` to a `Gen` enum reduces identically.

    Every other route to a codec is left UNRESOLVED rather than guessed; [codec_calls] reports those.
    """
    bound = set(CODEC_BOUND.findall(src))
    grew = True
    while grew:
        grew = False
        for name, body in CODEC_CHOICE.findall(src):
            ids = set(re.findall(r"[A-Za-z_]\w*", CHOICE_COND.sub(" ", body))) - CHOICE_WORDS
            if name not in bound and ids and ids <= bound:
                bound.add(name)
                grew = True
    return bound


def codec_escapes(src: str) -> list[str]:
    """Non-private declarations that hand a codec out, letting one reach a file that never names it.

    Scanning only files that name `WhoopCodec` is sound while this is empty, so it is checked rather
    than assumed. Two rules, because [codec_receivers] alone reads only expression bodies: a bound
    receiver that is not `private`, and any declaration whose RETURN TYPE names the codec. Kotlin
    requires that annotation on a block body, so `fun pick(g: Gen): WhoopCodec { return … }` and
    `val current: WhoopCodec get() { … }` - both invisible to the inference - are caught by the second.

    Residual, stated rather than claimed away: an inferred expression body whose result is a codec
    inside a container (`val all = listOf(gen5, gen4)`) is bound by neither rule.
    """
    typed = {n for rx in CODEC_TYPED for n, ty in rx.findall(src) if "WhoopCodec" in ty}
    return sorted(
        n for n in codec_receivers(src) | typed
        if not re.search(rf"^[ \t]*(?:\w+[ \t]+)*private[ \t]+(?:\w+[ \t]+)*(?:fun|val|var)[ \t]+{re.escape(n)}\b",
                         src, re.M)
    )


def codec_calls(names: list[str]) -> tuple[set[str], dict[str, list[str]], list[str]]:
    """Which `WhoopCodec` methods hand-written Kotlin calls, what it could not resolve, what escapes.

    A method counts only when its RECEIVER is known to be a codec. Four matchers answered this
    question by name alone and three published the wrong number: a bare `clientHello` found
    `DeviceFamily.clientHello`, `.feed(` found `reassembler.feed(bytes)`, and binding only the
    directly-constructed identifiers missed every call routed through `codec(isGen5)`.

    A regex has no types, so this cannot be exact. What it can be is one-sided and loud: an unresolved
    receiver on a name that IS a codec method is returned as unresolved, never counted either way, and
    the caller fails on it rather than publishing a guess.
    """
    wanted = {camel(n): n for n in names}
    called: set[str] = set()
    unresolved: dict[str, list[str]] = {}
    escaped: list[str] = []
    for src in kotlin_files():
        if "WhoopCodec" not in src:
            continue
        src = strip_comments(src)
        # No space before the paren: a KDoc sentence writes "WhoopCodec (decode history/live/…)",
        # which is prose, not a construction.
        if re.search(r"\bWhoopCodec\(", src):
            called.add("new")
        bound = codec_receivers(src)
        escaped += codec_escapes(src)
        for recv, method in CODEC_CALL.findall(src):
            if method not in wanted:
                continue
            if recv in bound:
                called.add(wanted[method])
            else:
                unresolved.setdefault(wanted[method], []).append(f"{recv}.{method}")
    return called, unresolved, sorted(set(escaped))


def codec_doc_lists(text: str) -> tuple[set[str], set[str]]:
    """The two halves `data-flow.md` NAMES, rather than the totals it used to count.

    Every wrong answer this claim published was an integer that moved by one or two and read as
    ordinary prose. A set difference names the symbol that appeared or vanished, and `feed` sitting
    in a list of methods the app calls is legible to anyone who has read `RustCodec.kt`.
    """
    def half(label: str) -> set[str]:
        m = re.search(rf"\*\*{label}\*\*(.*?)\n\n", text, re.S)
        return set(re.findall(r"`(\w+)`", m.group(1))) if m else set()

    return half("Called from Kotlin"), half("Not called")


def surface_table_rows() -> set[str]:
    """Rows in `data-flow.md`'s own surface section, one per `| \\`name\\` | what |`.

    Scoped to that section: the Kotlin-stays table below it has the same row shape, so an unscoped
    scan reads `ReadinessEngine` as a claimed export.
    """
    t = doc("data-flow.md")
    if "## The whoop-rs surface" not in t:
        return set()
    section = t[t.index("## The whoop-rs surface"):]
    section = section[: section.index("\n## ", 1)]
    return set(re.findall(r"^\| `(\w+)` \|", section, re.M))


def sleep_api_called_beyond_analyze() -> int:
    """`sleep_api` exports the app calls, less `analyze_sleep` itself.

    `sleep.md` called the app a frontend over "the single `analyzeSleep` FFI" while it drives the
    main-night family, the single-span restage and the debt/regularity/nap doors as well.
    """
    t = (ROOT / "crates" / "whoop-ffi" / "src" / "sleep_api.rs").read_text(encoding="utf-8")
    names = re.findall(r"#\[uniffi::export\][^\n]*\n(?:\s*//[^\n]*\n)*\s*pub fn (\w+)", t)
    reached = set(re.findall(r"uniffi\.whoop_ffi\.(\w+)", kotlin_blob()))
    return sum(1 for n in names if n != "analyze_sleep" and camel(n) in reached)


def exports_in_no_doc(names: list[str]) -> list[str]:
    """Exports named in neither shipped surface document.

    `architecture.md` said `data-flow.md` tables all of them. It tables 47, and a quarter of the
    surface is described nowhere at all — invisible to a count of the exports themselves.
    """
    text = doc("data-flow.md") + doc("algorithms.md") + doc("sleep.md")
    return sorted(n for n in names if not re.search(rf"\b{re.escape(n)}\b", text))


def exports_via_dead_wrapper(names: list[str]) -> list[str]:
    """Exports whose every Kotlin crossing sits in a wrapper nothing else names, main or test.

    `free_fns_called` asks whether a hand-written file crosses the FFI, which a wrapper satisfies on
    its own. `architecture.md` makes the stronger claim that the app CALLS each one, and a wrapper
    with no caller does not. Scans main AND test, so a parity test counts as a consumer; the
    qualified crossing is cut before counting or every wrapper looks referenced by itself.
    """
    src = KOTLIN.parent.parent if KOTLIN.name == "java" else KOTLIN
    if not src.is_dir():
        return []
    bodies = {
        p: strip_comments(p.read_text(encoding="utf-8", errors="replace"))
        for p in src.rglob("*.kt") if p.name != "whoop_ffi.kt"
    }
    unqualified = re.sub(r"uniffi\.whoop_ffi\.\w+", "", "\n".join(bodies.values()))

    def wrapper_at(text: str, idx: int) -> str | None:
        """Name of the innermost `val`/`var`/`fun` declaration the crossing at `idx` belongs to."""
        head = text[max(0, text.rfind("\n", 0, idx) - 300): idx + 1]
        found = list(re.finditer(r"(?:val|var|fun)\s+(\w+)", head))
        return found[-1].group(1) if found else None

    out = []
    for n in names:
        wrappers = {
            w
            for t in bodies.values()
            for m in re.finditer(rf"uniffi\.whoop_ffi\.{camel(n)}\b", t)
            if (w := wrapper_at(t, m.start()))
        }
        if wrappers and all(len(re.findall(rf"\b{w}\b", unqualified)) <= 1 for w in wrappers):
            out.append(n)
    return sorted(out)


def family_branches() -> int:
    """Non-test `if family == Family::GenX` sites outside `framing`/`family`.

    `architecture.md` claimed every per-generation wire difference is data on a `HeaderSpec` matched
    in one place. `HeaderSpec` carries the frame HEADER shape only, so opcodes, the GEN5 event
    residual and the GEN5-only record versions each branch at their own site — and unlike a `match`
    on the closed enum, an `if ==` would compile silently against a third generation.

    NON-TEST is enforced by PATH, not only by `#[cfg(test)]`. An integration test under `crates/*/tests/`
    carries no `#[cfg(test)]` marker, so the whole file counted and three branches in
    `sensitivity_decode.rs` were billed to the shipped code. That was the entire difference between the
    11 this used to return and the 8 the document correctly stated.
    """
    n = 0
    for p in rs_files("crates"):
        if p.stem in {"framing", "family"} or {"tests", "examples", "benches"} & set(p.parts):
            continue
        t = p.read_text(encoding="utf-8")
        cut = t.find("#[cfg(test)]")
        n += len(re.findall(r"(?:!=|==)\s*(?:crate::)?(?:family::)?Family::Gen[45]", t[:cut] if cut >= 0 else t))
    return n


def kotlin_backlog_rows() -> int | None:
    """Rows in noop-tan's own Kotlin-still-owns-maths table — the border's OTHER direction.

    `algorithms.md` claimed no metric the frontend still computes itself while that table listed 17,
    a contradiction between two shipped documents that neither gate could see: both stop at this
    repo. The table itself is re-derived against the Kotlin by `audit_kotlin_algorithms.py`, so
    pinning the sentence to the table is the missing edge, not a second count of the same thing.

    The slice stops at the NEXT `## ` heading, not at the re-derived footer far below it. Ending at the
    footer swept in every table between, so the five Rust exports tabled under the optical-quality
    section were counted as Kotlin files still owning maths and the answer read 25 for a table of 20.
    """
    doc = WHOOP / "noop-wt-tan" / "docs" / "ALGORITHMS.md"
    if not doc.is_file():
        return None
    t = doc.read_text(encoding="utf-8")
    if "## Remaining" not in t or "**The counts above are re-derived**" not in t:
        return -1
    start = t.index("## Remaining")
    nxt = t.find("\n## ", start + 1)
    section = t[start: nxt if nxt > 0 else t.index("**The counts above are re-derived**")]
    return len(re.findall(r"^\| `([\w./]+)`", section, re.M))


def kappa_targets() -> dict[str, str]:
    """Each cohort's asserted kappa, straight out of the parity gate's own assertions."""
    t = (ROOT / "crates" / "physio-algo" / "tests" / "dataset_parity.rs").read_text(encoding="utf-8")
    return dict(re.findall(r'run_dataset\("([\w-]+)"\)[\s\S]{0,400}?off target (\d\.\d+)"', t))


def named_sets() -> set[str]:
    t = (ROOT / "crates" / "physio-algo" / "tests" / "dataset_parity.rs").read_text(encoding="utf-8")
    block = t[t.index("const SETS"):]
    return set(re.findall(r'\("([\w-]+)",', block[: block.index("];")]))


def on_disk_sets() -> set[str]:
    return {p.name for p in FIXTURES.iterdir() if p.is_dir()} if FIXTURES.is_dir() else set()


def cargo_counts(args: list[str]) -> tuple[int, int]:
    """(passed, ignored) as the run itself reports them.

    IGNORED is read from the run rather than scanned for, because a static `#[ignore]` count is a third
    number again: an attribute behind a `cfg` or on an item the harness never registers is in the source
    and not in the summary line, and README's sentence quotes the summary line.
    """
    r = subprocess.run(["cargo", "test", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cargo test {' '.join(args)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return (
        sum(int(m) for m in re.findall(r"^test result: ok\. (\d+) passed", r.stdout, re.M)),
        sum(int(m) for m in re.findall(r"passed; \d+ failed; (\d+) ignored", r.stdout)),
    )


def command_opcodes() -> int:
    """Command opcode constants in `whoop-protocol/src/command.rs` (not the FORBIDDEN/DESTRUCTIVE lists)."""
    t = (ROOT / "crates" / "whoop-protocol" / "src" / "command.rs").read_text(encoding="utf-8")
    return len(re.findall(r"^pub const [A-Z0-9_]+: u8 = \d+;", t, re.M))


def ffi_touching_files() -> list[str]:
    """Hand-written Kotlin files under `main/` that reach `uniffi.whoop_ffi` directly.

    `data-flow.md` called `RustScores.kt` the single adapter. It is the largest by far, but it is not
    the only one, and a claim of singularity is worth a count rather than a reading.
    """
    main = KOTLIN.parent / "main" / "java" if (KOTLIN.parent / "main").is_dir() else KOTLIN
    if not main.is_dir():
        return []
    return sorted(
        p.name for p in main.rglob("*.kt")
        if p.name != "whoop_ffi.kt" and "uniffi.whoop_ffi" in p.read_text(encoding="utf-8", errors="replace")
    )


def codec_naming_files() -> list[str]:
    """Hand-written Kotlin files that name `WhoopCodec`, which is the OTHER half of the scope argument.

    [codec_escapes] proves no holder hands a codec out; this proves there is only one holder. The
    document rests on both, and a second file naming the type but calling nothing would leave the
    escape check green while `the only file` quietly became false.
    """
    main = KOTLIN.parent / "main" / "java" if (KOTLIN.parent / "main").is_dir() else KOTLIN
    if not main.is_dir():
        return []
    return sorted(
        p.name for p in main.rglob("*.kt")
        if p.name != "whoop_ffi.kt" and "WhoopCodec" in p.read_text(encoding="utf-8", errors="replace")
    )


def physio_algo_orphans() -> list[str]:
    """`physio-algo` public free fns reached by no other crate and by nothing inside `physio-algo`.

    The Rust shape of the failure that cost noop-tan three shipped capabilities: an item survives a
    green build and a green suite while its last caller is gone. Comments are stripped first, so a KDoc
    mention is not a use; the count includes each declaration, so a name at its own declaration count is
    referenced by nothing.

    FREE is column 0. Allowing leading whitespace swept in `impl` methods, and reported `Sweep::converged`
    as an orphan — a public method used thirteen times by `tests/ecg_sweep.rs`, which is outside the
    `src/` blob this scans by design. A method is not what the sentence claims, so it is not counted.
    """
    src = sorted((ROOT / "crates" / "physio-algo" / "src").rglob("*.rs"))
    def bare(p: Path) -> str:
        return "\n".join(line.split("//")[0] for line in p.read_text(encoding="utf-8").splitlines())
    declared: dict[str, int] = {}
    for p in src:
        for m in re.finditer(r"^pub fn (\w+)", bare(p), re.M):
            declared[m.group(1)] = declared.get(m.group(1), 0) + 1
    inside = "\n".join(bare(p) for p in src)
    outside = "\n".join(
        bare(p) for c in ("whoop-ffi", "whoopctl", "whoop-store", "whoop-client")
        for p in sorted((ROOT / "crates" / c / "src").rglob("*.rs"))
    )
    seen_out = set(re.findall(r"\b\w+\b", outside))
    return sorted(
        n for n, d in declared.items()
        if n not in seen_out and len(re.findall(rf"\b{re.escape(n)}\b", inside)) <= d
    )


def workspace_members() -> set[str]:
    """Crate directories under `crates/`, which is what the root Cargo.toml globs."""
    return {p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").exists()}


def readme_crate_rows() -> set[str]:
    """The crate names in README's Workspace table, one per `| \\`name\\` | role |` row."""
    t = doc("README.md")
    table = t[t.index("## Workspace"):]
    table = table[: table.index("\n## ", 1)]
    return set(re.findall(r"^\| `([a-z0-9-]+)` \|", table, re.M))


def kotlin_tree_dirty() -> list[str]:
    """Uncommitted Kotlin in the OTHER repo, which every cross-repo claim here silently reads.

    Four claims are derived from `noop-wt-tan`'s working tree, so another agent's half-finished edit
    moves them - one adding an import took "Kotlin files crossing the FFI" from 16 to 17 while the
    committed answer was still 16. Not a failure and not counted as one, but a red with this printed
    under it is a red nobody has to diagnose twice.
    """
    repo = KOTLIN.parents[4]
    if not repo.is_dir():
        return []
    r = subprocess.run(["git", "status", "--porcelain", "--", "*.kt"], cwd=repo, capture_output=True, text=True)
    return [ln[3:] for ln in r.stdout.splitlines()] if r.returncode == 0 else []


def rust_tree_dirty() -> list[str]:
    """Uncommitted Rust in THIS repo, which every count derived from `crates/` silently reads.

    The Kotlin side has had this note since a stray import moved a published number. This repo had none,
    and the same thing happened here: an unrelated `pack` module arriving mid-session as two untracked
    files took the workspace suite from 911 to 926 and `whoop-ffi` from 27 to 30, with nothing in the
    output to say the tree had grown a crate's worth of tests that no commit carries.
    """
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "crates"],
                       cwd=ROOT, capture_output=True, text=True)
    return [ln[3:] for ln in r.stdout.splitlines()] if r.returncode == 0 else []


def doc(name: str) -> str:
    """A shipped document by name. `docs/` first, then the repo root — README.md and AGENTS.md are
    shipped too, and were pinned by nothing until they had drifted a deleted crate and three counts."""
    p = DOCS / name
    return (p if p.exists() else ROOT / name).read_text(encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tests", action="store_true", help="skip the two cargo runs")
    a = ap.parse_args()

    exports = ffi_exports()
    methods = codec_methods()
    kappas = kappa_targets()

    # (document, what the claim is, regex with one capture, the derived truth)
    claims: list[tuple[str, str, str, object]] = [
        ("algorithms.md", "sleep-dataset gates", r"\*\*(\w+)\*\* sleep-dataset tests read external fixtures",
         SPELLED.get(dataset_gate_attrs(), str(dataset_gate_attrs()))),
        ("algorithms.md", "FFI free-fn exports", r"all (\d+) exported functions have a Kotlin", len(exports)),
        # "files", not "engines": the table's four decode leaks are not engines, and calling them that is
        # how a 20-row table got quoted as 17.
        ("algorithms.md", "Kotlin still owning maths", r"\*\*(\d+) Kotlin files still carry maths",
         kotlin_backlog_rows()),
        ("sleep.md", "physio-algo tests", r"\*\*(\d+) `physio-algo` tests", None),
        ("sleep.md", "whoop-ffi tests", r"`physio-algo` tests \+ (\d+) `whoop-ffi` tests", None),
        ("architecture.md", "generation branches", r"there are \*\*(\d+)\*\* of those outside", family_branches()),
        ("data-flow.md", "FFI free-fn exports", r"\*\*(\d+) exported functions, all", len(exports)),
        ("data-flow.md", "FFI exports called", r"exported functions, all (\d+) called", free_fns_called(exports)),
        ("data-flow.md", "codec methods", r"a further \*\*(\d+) methods\*\*", len(methods)),
        ("README.md", "command opcodes", r"CRC32\), (\d+) command opcodes", command_opcodes()),
        ("algorithms.md", "physio-algo orphans", r"\*\*(\d+) public\s+function[s]? in the crate (?:is|are) reached by no other crate\*\*",
         len(physio_algo_orphans())),
        ("data-flow.md", "Kotlin files crossing the FFI",
         r"\*\*(\d+) hand-written Kotlin files under `main/` reach `uniffi\.whoop_ffi` directly\*\*",
         len(ffi_touching_files())),
        ("sleep.md", "sleep_api exports the app calls",
         r"\*\*(\d+) further `sleep_api` exports\*\*", sleep_api_called_beyond_analyze()),
        ("data-flow.md", "surface-table rows", r"tables below carry \*\*(\d+)\*\* of them",
         len(surface_table_rows() & set(exports))),
        # Both halves of "N of the M" are captured: writing the shortfall while leaving a stale total
        # beside it is the same drift one number later.
        ("data-flow.md", "exports in no shipped doc", r"\*\*(\d+ of the \d+) appear in no shipped document\*\*",
         f"{len(exports_in_no_doc(exports))} of the {len(exports)}"),
        ("architecture.md", "exports reached only via a dead wrapper",
         r"(\w+) of them\s+\(`spo2_rolling_reading`\) only through a Kotlin wrapper the app never calls",
         "one" if len(exports_via_dead_wrapper(exports)) == 1 else str(exports_via_dead_wrapper(exports))),
    ]

    if not a.no_tests:
        pa, _ = cargo_counts(["-p", "physio-algo"])
        ws, ws_ignored = cargo_counts(["--workspace"])
        claims += [
            ("algorithms.md", "physio-algo tests", r"carries \*\*(\d+) unit tests\*\*", pa),
            ("algorithms.md", "workspace tests", r"workspace runs \*\*(\d+)\*\*", ws),
            ("README.md", "workspace tests", r"`cargo test --workspace` \((\d+) passed", ws),
            ("README.md", "ignored in that run", r"passed, (\d+) `#\[ignore\]`d\)", ws_ignored),
        ]
        for i, c in enumerate(claims):
            if c[0] == "sleep.md" and c[1] == "physio-algo tests":
                claims[i] = (*c[:3], pa)
            if c[0] == "sleep.md" and c[1] == "whoop-ffi tests":
                claims[i] = (*c[:3], cargo_counts(["-p", "whoop-ffi"])[0])

    bad = 0
    checked = 0
    for name, what, pattern, truth in claims:
        if truth is None:
            print(f"  SKIP   {name:<16} {what:<26} (needs a cargo run; re-run without --no-tests)")
            continue
        m = re.search(pattern, doc(name))
        checked += 1
        if not m:
            print(f"  NOPIN  {name:<16} {what:<26} the sentence this gate pins is GONE from the document")
            bad += 1
            continue
        said, is_ = m.group(1), str(truth)
        if said.lower() != is_.lower():
            print(f"  WRONG  {name:<16} {what:<26} document says {said}, source says {is_}")
            bad += 1
        else:
            print(f"  ok     {name:<16} {what:<26} {is_}")

    # README's crate table, both directions. A count would not have caught this: the table carried a
    # `whoop-metrics` row for a crate deleted long ago, in the same repo whose CLAUDE.md says it does not
    # exist, and a new crate landing with no row is the same failure the other way round.
    members, rows = workspace_members(), readme_crate_rows()
    checked += 1
    if rows != members:
        ghost, missing = sorted(rows - members), sorted(members - rows)
        print(f"  WRONG  {'README.md':<16} {'crate table':<26} rows for no crate: {ghost}; crates with no row: {missing}")
        bad += 1
    else:
        print(f"  ok     {'README.md':<16} {'crate table':<26} {len(rows)} rows == {len(members)} crates")

    # `WhoopCodec`'s methods are object methods, so no namespace qualifies them the way
    # `uniffi.whoop_ffi.<name>` qualifies a free function, and four matchers in a row answered the
    # question with a namesake or missed a route. The document NAMES both halves instead of counting
    # them, so a disagreement says which symbol moved - `feed` sitting among the called ones is
    # legible where 18 against 16 was not.
    said_called, said_not = codec_doc_lists(doc("data-flow.md"))
    called, unresolved, escaped = codec_calls(methods)
    checked += 1
    if not said_called or not said_not:
        print(f"  NOPIN  {'data-flow.md':<16} {'codec method lists':<26} "
              f"the **Called from Kotlin** / **Not called** list this gate reads is GONE")
        bad += 1
    elif (said_called | said_not) != set(methods) or (said_called & said_not):
        stray = sorted((said_called | said_not) - set(methods))
        missing = sorted(set(methods) - said_called - said_not)
        print(f"  WRONG  {'data-flow.md':<16} {'codec method lists':<26} "
              f"the two lists do not partition the {len(methods)} methods: "
              f"named but no such method: {stray}; in neither list: {missing}; in both: {sorted(said_called & said_not)}")
        bad += 1
    elif said_called != called:
        print(f"  WRONG  {'data-flow.md':<16} {'codec method lists':<26} "
              f"listed as called but no Kotlin call found: {sorted(said_called - called)}; "
              f"called but listed as not: {sorted(called - said_called)}")
        bad += 1
    else:
        print(f"  ok     {'data-flow.md':<16} {'codec method lists':<26} "
              f"{len(said_called)} called, {len(said_not)} not, partitioning {len(methods)}")

    # The extractor's own blind spot, made loud. A regex has no types, so a receiver it cannot resolve
    # is reported rather than counted - and it fails only when that ambiguity could flip a published
    # name, which is exactly what `reassembler.feed(bytes)` did when it published 18.
    checked += 1
    ambiguous = sorted(site for n, sites in unresolved.items() if n in said_not for site in sites)
    if escaped or ambiguous:
        print(f"  WRONG  {'RustCodec.kt':<16} {'codec receiver resolution':<26} "
              f"receivers a holder publishes: {escaped}; "
              f"calls it cannot attribute, on names published as uncalled: {ambiguous}")
        bad += 1
    else:
        print(f"  ok     {'RustCodec.kt':<16} {'codec receiver resolution':<26} "
              f"every codec-named call bound to a receiver, none escaping its file")

    # The other half of the same argument. Reading one file is sound because nothing escapes it AND
    # because it is the only one — and only the first half had a check, so a second holder that called
    # nothing would leave this green while the document's "the only file" became false.
    holders = codec_naming_files()
    checked += 1
    if holders != ["RustCodec.kt"]:
        print(f"  WRONG  {'data-flow.md':<16} {'the only codec holder':<26} "
              f"hand-written files naming WhoopCodec: {holders}")
        bad += 1
    else:
        print(f"  ok     {'data-flow.md':<16} {'the only codec holder':<26} RustCodec.kt, and no other")

    # The surface table's ghost direction. Its shortfall is a counted claim above (the table is a map,
    # not an API reference), but a row naming an export that no longer exists reads as ordinary prose.
    ghost_rows = sorted(surface_table_rows() - set(exports))
    checked += 1
    if ghost_rows:
        print(f"  WRONG  {'data-flow.md':<16} {'surface-table ghosts':<26} rows for no export: {ghost_rows}")
        bad += 1
    else:
        print(f"  ok     {'data-flow.md':<16} {'surface-table ghosts':<26} {len(surface_table_rows())} rows, all exported")

    # Each cohort kappa must be written the SAME way everywhere it appears. A document spells the four
    # cohorts across one comma-separated row, so a fragment is the unit: any decimal sharing a fragment
    # with a cohort name is claiming to BE that cohort's kappa.
    text = doc("algorithms.md")
    fragments = re.split(r"[,|\n;]|(?<=[A-Za-z)])\.\s", text)
    for cohort, target in sorted(kappas.items()):
        checked += 1
        others: set[str] = set()
        for frag in fragments:
            if re.search(re.escape(cohort), frag, re.I):
                others |= set(re.findall(r"\d\.\d{2,3}", frag))
        stale = {v for v in others if v != target and not target.startswith(v)}
        if target not in text:
            print(f"  WRONG  {'algorithms.md':<16} {'kappa ' + cohort:<26} gate asserts {target}, document never states it")
            bad += 1
        elif stale:
            print(f"  WRONG  {'algorithms.md':<16} {'kappa ' + cohort:<26} gate asserts {target}, document also says {sorted(stale)}")
            bad += 1
        else:
            print(f"  ok     {'algorithms.md':<16} {'kappa ' + cohort:<26} {target}")

    # `examples/common/mod.rs` claims it owns every fixture-corpus reader. That sentence has been false
    # twice, so it gets a command: no harness but the declared exception may touch the filesystem itself.
    ex = ROOT / "crates" / "physio-algo" / "examples"
    allowed = {"verify_backup.rs"}
    rogue = sorted(
        p.name
        for p in ex.glob("*.rs")
        if p.name not in allowed and re.search(r"fs::read_to_string|fs::read_dir|File::open", p.read_text(encoding="utf-8"))
    )
    checked += 1
    if rogue:
        print(f"  WRONG  {'common/mod.rs':<16} {'owns every corpus reader':<26} harnesses reading directly: {rogue}")
        bad += 1
    else:
        print(f"  ok     {'common/mod.rs':<16} {'owns every corpus reader':<26} {len(list(ex.glob('*.rs')))} harnesses, {sorted(allowed)} excepted")

    # Every symbol `sleep.md`'s module table names must exist in `physio-algo`. A count check cannot see
    # this: the table named `bridge_sparse_sleep` (the code calls it `bridge_sleep_gap`) and `RespSample`
    # (never existed), and both read as ordinary prose. Same class as the crate the architecture diagram
    # was built on that was not a member of the workspace.
    text = doc("sleep.md")
    checked += 1
    named, files, absent = set(), set(), []
    if "## Modules" not in text:
        print(f"  NOPIN  {'sleep.md':<16} {'module table symbols':<26} the section this gate reads is GONE")
        bad += 1
    else:
        table = text[text.index("## Modules"):]
        table = table[: table.index("\n## ", 1)]
        src = "\n".join(p.read_text(encoding="utf-8") for p in rs_files("crates/physio-algo/src"))
        for tok in re.findall(r"`([^`]+)`", table):
            if tok.endswith(".rs"):
                files.add(tok)
                continue
            if " " in tok:
                continue
            for part in re.split(r"::|\.", tok):
                if re.fullmatch(r"[A-Za-z_]\w*", part) and part not in {"crate", "stats"}:
                    named.add(part)
        absent = [n for n in sorted(named) if not re.search(rf"\b{re.escape(n)}\b", src)]
    if not named:
        pass
    elif absent:
        print(f"  WRONG  {'sleep.md':<16} {'module table symbols':<26} named but absent from physio-algo: {absent}")
        bad += 1
    else:
        print(f"  ok     {'sleep.md':<16} {'module table symbols':<26} {len(named)} named, all present")

    # The same table's FILE column drifts in two directions the symbol check above cannot see, because it
    # skips any token ending `.rs`. A row naming a file that is not there reads as ordinary prose, and a
    # module declared in `sleep/mod.rs` with no row is invisible by construction - which is how `params.rs`
    # stayed undocumented. Both sides are pinned against `mod.rs`, not against a count.
    sleep_dir = ROOT / "crates" / "physio-algo" / "src" / "sleep"
    body = re.sub(r"#\[cfg\(test\)\]\s*mod \w+;", "", (sleep_dir / "mod.rs").read_text(encoding="utf-8"))
    declared = set(re.findall(r"^(?:pub )?mod (\w+);", body, re.M))
    if files or declared:
        checked += 1
        phantom = sorted(f for f in files if not (sleep_dir / f).is_file())
        unlisted = sorted(f"{m}.rs" for m in declared if f"{m}.rs" not in files)
        if phantom or unlisted:
            print(f"  WRONG  {'sleep.md':<16} {'module table files':<26} "
                  f"named but not on disk: {phantom}; declared in mod.rs but unlisted: {unlisted}")
            bad += 1
        else:
            print(f"  ok     {'sleep.md':<16} {'module table files':<26} {len(files)} rows == {len(declared)} modules")

    # Every fixture set on disk must be named in the parity sheet's matrix.
    disk, named = on_disk_sets(), named_sets()
    if disk:
        checked += 1
        missing = sorted(disk - named)
        if missing:
            print(f"  WRONG  {'dataset_parity.rs':<16} {'fixture-set matrix':<26} on disk but unnamed: {missing}")
            bad += 1
        else:
            print(f"  ok     {'dataset_parity.rs':<16} {'fixture-set matrix':<26} {len(named)} sets, none unnamed")
    else:
        print(f"  SKIP   {'dataset_parity.rs':<16} {'fixture-set matrix':<26} fixture root absent")

    dirty = kotlin_tree_dirty()
    if dirty:
        print(f"\n  NOTE   noop-wt-tan has {len(dirty)} uncommitted Kotlin file(s); the four cross-repo claims\n"
              f"         above read the working tree, not the commit: {dirty}")

    rust_dirty = rust_tree_dirty()
    if rust_dirty:
        print(f"\n  NOTE   whoop-rs has {len(rust_dirty)} uncommitted file(s) under crates/; every count derived\n"
              f"         from the source — the two cargo runs included — reads the working tree, not the\n"
              f"         commit: {rust_dirty}")

    print(f"\n{checked} claims checked | {bad} disagree with the source")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
