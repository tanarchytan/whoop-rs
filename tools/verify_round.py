#!/usr/bin/env python3
"""The mechanical half of the verification round, run after every plan step.

Four of the defects found on 2026-08-31/09-01 were mechanical and would have been caught by a
script: an orphaned public field nothing read, a constant defined twice, a recorded test count that
had gone stale twice in two days, and a dataset column the loader never opened. Those are here.

The other half CANNOT be automated and lives in whoop-rs/CLAUDE.md under "Verification round".
Its own tests are tools/test_verify_round.py, which also fails if a new check arrives untested.
Running this green does not mean the round passed; it means the mechanical part did.

    python whoop-rs/tools/verify_round.py            # everything, incl. the ignored suite
    python whoop-rs/tools/verify_round.py --fast     # skip the cargo runs
    python whoop-rs/tools/verify_round.py --self-test

Exit 0 all clear, 1 a FAIL, 2 the script could not run.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / "whoop-rs"
CRATES = RS / "crates"

FAIL, WARN, OK = "FAIL", "warn", "ok"
results = []


def say(check, status, detail=""):
    results.append((check, status, detail))


def cargo(args):
    p = subprocess.run(
        ["cargo", *args], cwd=RS, capture_output=True, text=True, errors="replace"
    )
    return p.stdout + p.stderr


def rs_files(where):
    return [p for p in where.rglob("*.rs") if "target" not in p.parts]


# The parsing and comparing are split out from the subprocess calls ON PURPOSE: a check whose
# failure path has never run is not a gate. These are pure, so `--self-test` can feed each one a
# fabricated defect and require the right verdict without a two-minute cargo run.

def parse_test_results(text):
    rows = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    return (
        sum(int(r[0]) for r in rows),
        sum(int(r[1]) for r in rows),
        sum(int(r[2]) for r in rows),
        len(rows),
    )


def compare_counts(passed, claims):
    """`claims` is [(label, recorded_or_None)]. A recorded count is a dated measurement."""
    for label, recorded in claims:
        if recorded is None:
            say(f"count in {label}", WARN, "no recorded count found to check")
        elif recorded != passed:
            say(f"count in {label}", FAIL, f"says {recorded}, measured {passed}")
        else:
            say(f"count in {label}", OK, str(passed))


def count_clippy(text):
    return len([ln for ln in text.splitlines() if re.match(r"^(warning|error)(\[|:)", ln)])


def unread_columns(header, src):
    cols = [c.strip().strip('"') for c in header.split(",")]
    return cols, [c for c in cols if f'"{c}"' not in src]


# ---------------------------------------------------------------- 1. counts are fresh

COUNT_CLAIMS = [
    (RS / "CLAUDE.md", re.compile(r"# (\d{3,5}) passed, (\d+) failed, (\d+) #\[ignore\]d")),
    (RS / "docs" / "sleep.md", re.compile(r"\*\*(\d{3,5}) workspace tests")),
]


def check_counts(fast):
    """A recorded test count is a measurement with a date on it, and it goes stale silently."""
    if fast:
        say("test counts fresh", WARN, "skipped (--fast)")
        return
    passed, failed, ignored, n = parse_test_results(cargo(["test"]))
    if n == 0:
        say("suite green", FAIL, "no `test result` lines - did cargo test run at all?")
        return
    if failed:
        say("suite green", FAIL, f"{failed} failed")
        return
    say("suite green", OK, f"{passed} passed, {ignored} ignored")

    claims = []
    for path, pat in COUNT_CLAIMS:
        if not path.exists():
            continue
        m = pat.search(path.read_text(encoding="utf-8"))
        claims.append((path.name, int(m.group(1)) if m else None))
    compare_counts(passed, claims)


def check_ignored_suite(fast):
    """The 52 #[ignore]d tests: cohort gates, negative controls, source cross-checks.

    `cargo test` does not reach them and nothing else runs them, which is how two gates went stale
    for days. Paths are ABSOLUTE and derived here, because cargo runs each test binary with its own
    PACKAGE as the working directory -- a relative `../whoop-firmware` resolves inside
    crates/whoop-protocol and the firmware gate fails with NotFound, which is what happened.
    """
    if fast:
        say("ignored suite", WARN, "skipped (--fast) - this is where stale gates hide")
        return
    zbin = ROOT / "whoop-firmware" / ".zbin-extract"
    capture = ROOT / "whoop-data" / "own-data" / "noop-raw-capture-260729-0928.jsonl"
    missing = [str(p) for p in (zbin, capture) if not p.exists()]
    if missing:
        say("ignored suite", FAIL, "fixture missing, so the gates cannot run: " + ", ".join(missing))
        return
    env = {"WHOOP_ZBIN_DIR": str(zbin), "WHOOP_CAPTURE": str(capture)}
    import os
    p = subprocess.run(
        ["cargo", "test", "--release", "--workspace", "--", "--ignored"],
        cwd=RS, capture_output=True, text=True, errors="replace", env={**os.environ, **env},
    )
    report_ignored(p.stdout + p.stderr)


def report_ignored(out):
    passed, failed, _, n = parse_test_results(out)
    named = re.findall(r"^\s{4}(\S+::\S+)$", out, re.M)
    if n == 0:
        say("ignored suite", FAIL, "no `test result` lines - it did not run")
        return
    say("ignored suite", OK if failed == 0 else FAIL,
        f"{passed} passed, {failed} failed" + (f" -- {', '.join(named[:4])}" if failed else ""))


def check_clippy(fast):
    if fast:
        say("clippy clean", WARN, "skipped (--fast)")
        return
    n = count_clippy(cargo(["clippy", "--workspace", "--all-targets"]))
    say("clippy clean", OK if n == 0 else FAIL, f"{n} diagnostic(s)")


# ---------------------------------------------------------------- 2. one definition each

CONST_DEF = re.compile(r"^\s*(?:pub(?:\(\w+\))?\s+)?const\s+([A-Z][A-Z0-9_]{2,})\s*:", re.M)


def added_names(since):
    """Identifiers on lines this step ADDED, working tree included.

    Scope is the whole point. Run over the tree as a whole, the orphan check reports 380 items,
    nearly all of them the deliberately-unwired FFI surface `CLAUDE.md` says to keep -- and a
    report nobody reads catches nothing. Scoped to the step, it caught three in one pass.
    """
    p = subprocess.run(
        ["git", "diff", "--unified=0", since, "--", "crates"],
        cwd=RS, capture_output=True, text=True, errors="replace",
    )
    if p.returncode != 0:
        return None
    added = [ln[1:] for ln in p.stdout.splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    # `git diff` does not see UNTRACKED files, and a brand-new harness is untracked until it is
    # staged. Every line of one counts as added, or the step's newest file is the one unscanned --
    # which is how seven duplicated constants passed an after-step round on 2026-09-01.
    u = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "crates"],
                       cwd=RS, capture_output=True, text=True, errors="replace")
    for rel in u.stdout.split():
        f = RS / rel
        if f.suffix == ".rs" and f.exists():
            added.extend(f.read_text(encoding="utf-8", errors="replace").splitlines())
    return names_in(added)


def names_in(lines):
    """Public items, constants and pub struct fields declared on these lines."""
    names = set()
    for body in lines:
        names.update(PUB_ITEM.findall(body))
        names.update(CONST_DEF.findall(body))
        names.update(re.findall(r"^\s+pub ([a-z_][a-z0-9_]*):", body))
    return names


def check_duplicate_consts(scope):
    """COVERAGE_KEEP lived in two screens, so `COVERAGE` could quietly become two arms."""
    where = {}
    for f in rs_files(CRATES):
        for name in set(CONST_DEF.findall(f.read_text(encoding="utf-8", errors="replace"))):
            where.setdefault(name, []).append(f)
    dupes = {
        n: v for n, v in where.items()
        # Same name in two files is only a smell when they are not both test-local.
        if len(v) > 1 and len({p.name for p in v}) > 1 and (scope is None or n in scope)
    }
    if not dupes:
        say("no constant defined twice", OK)
        return
    lines = [f"{n}: " + ", ".join(p.relative_to(RS).as_posix() for p in v)
             for n, v in sorted(dupes.items())]
    say("no constant defined twice", WARN, f"{len(dupes)} shared name(s)\n      " + "\n      ".join(lines))


# ---------------------------------------------------------------- 3. nothing orphaned

PUB_ITEM = re.compile(r"^\s*pub (?:fn|const|struct|enum) ([A-Za-z_][A-Za-z0-9_]*)", re.M)


def body_and_tests(text):
    """Split a file at its test module: a reference only from `mod tests` is not a caller."""
    i = text.find("#[cfg(test)]")
    return (text, "") if i < 0 else (text[:i], text[i:])


def check_orphans(scope):
    """Structure::occupancy was written every segment and read by nothing but its own test.

    One tokenising pass over every file, then O(1) lookups. Searching per name per file instead is
    quadratic and takes minutes, which means it would not get run.
    """
    bs = chr(92)
    word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    files = {f: f.read_text(encoding="utf-8", errors="replace") for f in rs_files(CRATES)}
    # name -> set of files whose NON-TEST body mentions it, and name -> set of any file at all.
    in_body, anywhere = {}, {}
    for f, text in files.items():
        body, tests = body_and_tests(text)
        for tok in set(word.findall(body)):
            in_body.setdefault(tok, set()).add(f)
        for tok in set(word.findall(text)):
            anywhere.setdefault(tok, set()).add(f)

    orphans = []
    for f, text in files.items():
        if "examples" in f.parts:
            continue
        body, _ = body_and_tests(text)
        for name in set(PUB_ITEM.findall(body)):
            if scope is not None and name not in scope:
                continue
            # A caller is any OTHER file mentioning it. Its own test module does not count.
            if anywhere.get(name, set()) - {f}:
                continue
            # Or a second mention inside its own non-test body, beyond the definition line.
            # Word-bounded: a substring match would let a longer name count as a caller.
            pat = bs + "b" + re.escape(name) + bs + "b"
            if len(re.findall(pat, body)) > 1:
                continue
            orphans.append(f"{f.relative_to(RS).as_posix()}::{name}")
    if not orphans:
        say("no orphaned pub items", OK)
    else:
        head = f"{len(orphans)} with no caller outside their own file and tests"
        indent = chr(10) + "      "
        say("no orphaned pub items", WARN, head + indent + indent.join(sorted(orphans)[:25]))


# ---------------------------------------------------------------- 4. the dataset is fully read

def check_dataset_columns():
    """MESA's `Type` column sat unread, so RMSSD ran on intervals it is not defined on.

    Only corpora with a HEADER can be audited this way. The PSG fixtures under
    `sleep-benchmark/fixtures_multi_clean3` are headerless numeric CSVs, so there are no named
    columns to leave unread and the check is not applicable rather than silently passing. What it
    does NOT cover there is the fixture BUILDER dropping a column on the way in -- a different
    check against the source dataset, not written.
    """
    loaders = list((CRATES / "physio-algo" / "examples" / "common").glob("*.rs"))
    if not loaders:
        say("dataset columns read", WARN, "no loaders found")
        return
    fixtures = ROOT / "sleep-benchmark" / "fixtures_multi_clean3"
    if fixtures.is_dir():
        say("psg fixture columns", WARN,
            "headerless CSVs - not auditable this way; the builder is unchecked")
    csvs = {
        "mesa": ROOT / "whoop-data" / "datasets" / "mesa" / "annotations-rpoints",
    }
    for name, d in csvs.items():
        loader = next((p for p in loaders if p.stem == name), None)
        if loader is None or not d.is_dir():
            say(f"{name} columns read", WARN, "loader or corpus missing")
            continue
        sample = next(iter(sorted(d.glob("*.csv"))), None)
        if sample is None:
            say(f"{name} columns read", WARN, "no csv in corpus")
            continue
        header = sample.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        src = loader.read_text(encoding="utf-8", errors="replace")
        cols, unread = unread_columns(header, src)
        say(f"{name} columns read", OK if not unread else WARN,
            f"{len(cols) - len(unread)}/{len(cols)} named in the loader"
            + (f"; UNREAD: {', '.join(unread)}" if unread else ""))


# ---------------------------------------------------------------- 5. the research claims resolve

def report_evidence(out, rc):
    if "OK -- every citation resolves" in out:
        n = re.search(r"total: (\d+) reference citation", out)
        say("research evidence gate", OK, f"{n.group(1) if n else '?'} citations resolve")
    else:
        first = next((ln for ln in out.splitlines() if ln.strip().startswith("[")), f"exit {rc}")
        say("research evidence gate", FAIL, first[:120])


def check_evidence(fast):
    """Every cited number resolves to a retrieved artifact, and every measurement has a falsifier.

    Run manually after each step until now, which is the same as not being in the round at all.
    """
    script = Path(__file__).resolve().parent / "check_evidence.py"
    if fast or not script.exists():
        say("research evidence gate", WARN, "skipped (--fast)" if fast else "script missing")
        return
    p = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    report_evidence(p.stdout + p.stderr, p.returncode)


# ---------------------------------------------------------------- 6. line endings

def check_line_endings():
    """A file mixing both is how a search-and-replace silently applies to nothing."""
    mixed = []
    for f in rs_files(RS):
        b = f.read_bytes()
        if b.count(b"\r\n") and b.count(b"\n") != b.count(b"\r\n"):
            mixed.append(f.relative_to(RS).as_posix())
    say("no file mixes CRLF and LF", OK if not mixed else FAIL,
        ", ".join(mixed[:8]) if mixed else "")


SCOPED = [check_duplicate_consts, check_orphans]
GLOBAL = [check_dataset_columns, check_line_endings]


GREEN_RUN = "test result: ok. 40 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out"
RED_RUN = (
    "failures:\n    sleep::v2::tests::a_probe\n"
    "test result: FAILED. 39 passed; 1 failed; 3 ignored; 0 measured; 0 filtered out"
)


def probe(label, want, run):
    """Run one planted defect and require the stated verdict. Returns True if the check fired."""
    results.clear()
    run()
    hit = want in [s for _, s, _ in results]
    print(f"  {'caught ' if hit else 'MISSED '} {label}")
    return hit


def self_test():
    """EVERY check must fire on a planted defect. Three of seven were covered until 2026-09-01,
    and the four that were not included the counts and the ignored suite -- the two that matter
    most. The subprocess calls are split from the parsing so the failure paths can be reached."""
    target = CRATES / "physio-algo" / "src" / "verify_round_probe.rs"
    ok = True
    try:
        def plant(text=None, raw=None):
            target.write_bytes(raw) if raw is not None else target.write_text(text, encoding="utf-8")

        plant("pub const RARE_SHARE: f64 = 1.0;\n")
        ok &= probe("a constant defined in two files", WARN, lambda: check_duplicate_consts(None))
        plant("pub fn zz_probe_never_called() {}\n")
        ok &= probe("an orphaned pub item", WARN, lambda: check_orphans(None))
        target.unlink(missing_ok=True)
        plant(raw=b"// a\r\n// b\n")
        ok &= probe("a file mixing CRLF and LF", FAIL, check_line_endings)
        target.unlink(missing_ok=True)

        # The four that had never been exercised, on fabricated tool output.
        ok &= probe("a recorded test count gone stale", FAIL,
                    lambda: compare_counts(1202, [("CLAUDE.md", 1201)]))
        ok &= probe("a missing recorded count", WARN,
                    lambda: compare_counts(1202, [("sleep.md", None)]))
        ok &= probe("a FAILING ignored suite", FAIL, lambda: report_ignored(RED_RUN))
        ok &= probe("an ignored suite that did not run", FAIL, lambda: report_ignored(""))
        ok &= probe("a green ignored suite passing", OK, lambda: report_ignored(GREEN_RUN))

        n = count_clippy("warning: unused variable `x`\n  --> a.rs:1\nerror[E0308]: mismatch\n")
        hit = n == 2
        print(f"  {'caught ' if hit else 'MISSED '} clippy diagnostics counted ({n}, want 2)")
        ok &= hit

        cols, unread = unread_columns('"epoch","Type","seconds"', 'find("epoch")?, find("seconds")?')
        ok &= probe("a research citation that does not resolve", FAIL,
                    lambda: report_evidence("[REF:ghost#q1] -- QUOTE NOT FOUND", 1))
        ok &= probe("a passing evidence gate", OK,
                    lambda: report_evidence("total: 49 reference citation(s)"
                                            + chr(10) + "OK -- every citation resolves to x", 0))
        hit = unread == ["Type"]
        print(f"  {'caught ' if hit else 'MISSED '} a dataset column the loader never opens {unread}")
        ok &= hit
    finally:
        target.unlink(missing_ok=True)
    print("\nself-test", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    if "--self-test" in sys.argv:
        return self_test()
    fast = "--fast" in sys.argv
    since = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--since=")), "HEAD~1")
    scope = added_names(since)
    check_counts(fast)
    check_ignored_suite(fast)
    check_clippy(fast)
    check_evidence(fast)
    for c in SCOPED:
        c(scope)
    for c in GLOBAL:
        c()

    print("VERIFICATION ROUND - mechanical half\n")
    worst = 0
    for name, status, detail in results:
        mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}[status]
        print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
        worst = max(worst, {OK: 0, WARN: 0, FAIL: 1}[status])
    print("\nThe JUDGEMENT half is not in here - see whoop-rs/CLAUDE.md 'Verification round'.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
