#!/usr/bin/env python3
"""Adversarial tests for the verification round itself.

    python whoop-rs/tools/test_verify_round.py

The round is the thing that decides whether a plan step is finished, so a check inside it that
cannot fail is worse than no check: it reports green and is believed. On 2026-09-01 exactly that
was true of four of its seven checks -- the recorded counts and the ignored suite among them, the
two that matter most -- because `--self-test` only planted defects for three.

Two kinds of test here, and the second is the one that matters.

  1. Every check FIRES on a planted defect, and stays quiet on a clean input. A check that always
     fires is as useless as one that never does, so both directions are asserted.
  2. `every_check_is_covered` enumerates the `check_*` functions by introspection and requires each
     to appear in COVERED. Adding an eighth check without a test fails this file. That is the guard
     against the 3-of-7 regression happening a second time, and nothing else in the round provides
     it.

The round shells out to cargo, so the verdict logic is split into pure functions there on purpose;
this exercises those. What it does NOT cover is the subprocess plumbing -- that a cargo invocation
is well-formed is proven by running the round for real, not here.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify_round", HERE / "verify_round.py")
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")


def verdicts(run) -> list[str]:
    """The statuses one check emits, with the shared result list isolated per call."""
    vr.results.clear()
    run()
    return [status for _, status, _ in vr.results]


# Each check maps to the function carrying its verdict, so the meta-test can demand coverage.
COVERED = {
    "check_counts": "compare_counts + report_empty_targets",
    "check_ignored_suite": "report_ignored",
    "check_clippy": "count_clippy",
    "check_evidence": "report_evidence",
    "check_dataset_columns": "unread_columns",
    "check_duplicate_consts": "planted probe file",
    "check_orphans": "planted probe file",
    "check_line_endings": "planted probe file",
    "check_eol_flips": "flipped tracked file",
}

EMPTY_DOC = ("     Running unittests src/lib.rs (t/a.exe)\n"
             "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
             "   Doc-tests foo\n\nrunning 0 tests\n\n"
             "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
EMPTY_FILE = ("     Running tests/empty.rs (t/b.exe)\n"
              "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")

GREEN = "test result: ok. 40 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out"
RED = ("failures:\n    sleep::v2::tests::a_probe\n"
       "test result: FAILED. 39 passed; 1 failed; 3 ignored; 0 measured; 0 filtered out")


def test_counts():
    check("a stale recorded count FAILS",
          verdicts(lambda: vr.compare_counts(1202, [("CLAUDE.md", 1201)])), [vr.FAIL])
    check("a matching count passes",
          verdicts(lambda: vr.compare_counts(1202, [("CLAUDE.md", 1202)])), [vr.OK])
    check("a missing count warns rather than passing",
          verdicts(lambda: vr.compare_counts(1202, [("sleep.md", None)])), [vr.WARN])
    # Off by one in either direction, because `!=` written as `<` would pass one of them.
    check("a count one HIGH also fails",
          verdicts(lambda: vr.compare_counts(1202, [("x", 1203)])), [vr.FAIL])
    check("totals sum across binaries", vr.parse_test_results(GREEN + "\n" + GREEN)[:3], (80, 0, 6))
    check("no test-result lines parses to nothing", vr.parse_test_results("")[3], 0)


def test_ignored_suite():
    check("a failing ignored suite FAILS", verdicts(lambda: vr.report_ignored(RED)), [vr.FAIL])
    check("a green ignored suite passes", verdicts(lambda: vr.report_ignored(GREEN)), [vr.OK])
    # David asked about "many things with zero tests": all 8 are DOC-test targets (no `///`
    # examples), which is a style choice. A real test FILE running nothing is a hole.
    check("a doc-test target with 0 tests is fine",
          verdicts(lambda: vr.report_empty_targets(EMPTY_DOC)), [vr.OK])
    check("a TEST FILE with 0 tests FAILS",
          verdicts(lambda: vr.report_empty_targets(EMPTY_FILE)), [vr.FAIL])
    # The case that actually happened: the command was wrong, so nothing ran. Silence is not green.
    check("an ignored suite that did not run FAILS",
          verdicts(lambda: vr.report_ignored("")), [vr.FAIL])
    vr.results.clear()
    vr.report_ignored(RED)
    check("and it names the failing test", "a_probe" in vr.results[0][2], True)


def test_clippy():
    check("clippy warnings and errors are both counted",
          vr.count_clippy("warning: unused `x`\n  --> a.rs:1\nerror[E0308]: mismatch\n"), 2)
    check("a clean clippy run counts zero", vr.count_clippy("    Finished in 0.7s\n"), 0)
    # Indented continuation lines are context, not diagnostics: counting them inflates the verdict.
    check("indented context lines are not diagnostics",
          vr.count_clippy("  warning: this is context\n   = help: try\n"), 0)


def test_evidence():
    check("an unresolved citation FAILS",
          verdicts(lambda: vr.report_evidence("[REF:ghost#q1] -- QUOTE NOT FOUND", 1)), [vr.FAIL])
    check("a resolving corpus passes",
          verdicts(lambda: vr.report_evidence(
              "total: 49 reference citation(s)\nOK -- every citation resolves to x", 0)), [vr.OK])


def test_dataset_columns():
    cols, unread = vr.unread_columns('"epoch","Type","seconds"', 'find("epoch")?, find("seconds")?')
    check("a column the loader never names is reported", unread, ["Type"])
    check("and the others are not", cols, ["epoch", "Type", "seconds"])
    _, none_unread = vr.unread_columns('"a","b"', 'find("a") find("b")')
    check("a fully-read header reports nothing", none_unread, [])
    # A bare substring would let `Typeless` satisfy `Type`; the quotes are what prevent that.
    _, still = vr.unread_columns('"Type"', 'find("TypeOfThing")')
    check("a longer namesake does not count as read", still, ["Type"])


def flipped(run=None):
    """Flip a TRACKED file to LF in the working tree, run one check, always restore.

    `check_eol_flips` compares the index form against the tree form, so an untracked probe cannot
    exercise it. `metrics.rs` is CRLF in the index, and it is the file the real flip happened to.
    """
    target = vr.CRATES / "physio-algo" / "src" / "sleep" / "metrics.rs"
    original = target.read_bytes()
    try:
        target.write_bytes(original.replace(b"\r\n", b"\n"))
        return verdicts(run or vr.check_eol_flips)
    finally:
        target.write_bytes(original)


def planted(probe_text=None, probe_bytes=None, run=None):
    """Write a probe file into the crate tree, run one check, always clean up."""
    target = vr.CRATES / "physio-algo" / "src" / "verify_round_probe.rs"
    try:
        if probe_bytes is not None:
            target.write_bytes(probe_bytes)
        else:
            target.write_text(probe_text, encoding="utf-8")
        return verdicts(run)
    finally:
        target.unlink(missing_ok=True)


def test_file_checks():
    check("a constant defined in two files warns",
          planted("pub const RARE_SHARE: f64 = 1.0;\n", run=lambda: vr.check_duplicate_consts(None)),
          [vr.WARN])
    check("an orphaned pub item warns",
          planted("pub fn zz_probe_never_called() {}\n", run=lambda: vr.check_orphans(None)),
          [vr.WARN])
    check("a file mixing CRLF and LF FAILS",
          planted(probe_bytes=b"// a\r\n// b\n", run=vr.check_line_endings), [vr.FAIL])
    # Clean tree: these must go quiet, or they would fire on everything and mean nothing.
    check("a consistently-CRLF file is fine",
          planted(probe_bytes=b"// a\r\n// b\r\n", run=vr.check_line_endings), [vr.OK])
    check("a pub item WITH a caller is not an orphan",
          planted("pub fn kappa4_probe() {}\n", run=lambda: vr.check_orphans({"nothing_added"})),
          [vr.OK])
    # `check_eol_flips` needs a TRACKED file -- an untracked probe has no index form to disagree
    # with -- so it flips a real one and restores the bytes. This is the defect the mixed-endings
    # check cannot see: a wholesale flip is internally consistent and buries the real diff.
    check("a file flipped WHOLESALE to LF FAILS", flipped(), [vr.FAIL])
    check("and the mixed-endings check stays GREEN on that same flip",
          flipped(run=vr.check_line_endings), [vr.OK])
    check("an unflipped tree is fine", verdicts(vr.check_eol_flips), [vr.OK])


def test_scope_is_the_step():
    """Unscoped, the orphan check reports ~380 items and nobody reads it. Scope is load-bearing."""
    out_of_scope = planted("pub fn zz_probe_never_called() {}\n",
                           run=lambda: vr.check_orphans({"some_other_name"}))
    check("an orphan OUTSIDE the step's scope is not reported", out_of_scope, [vr.OK])
    in_scope = planted("pub fn zz_probe_never_called() {}\n",
                       run=lambda: vr.check_orphans({"zz_probe_never_called"}))
    check("the same orphan INSIDE it is", in_scope, [vr.WARN])


def test_untracked_files_are_in_scope():
    """`git diff` cannot see an untracked file, and a new harness is untracked until staged. Seven
    duplicated constants passed an after-step round that way. Every line of one must count."""
    check("names_in reads a const, a pub fn and a pub field",
          vr.names_in(["const RARE_SHARE: f64 = 0.5;", "pub fn zz_new() {}", "    pub zz_field: u8,"]),
          {"RARE_SHARE", "zz_new", "zz_field"})
    probe = vr.CRATES / "physio-algo" / "src" / "verify_round_untracked_probe.rs"
    try:
        probe.write_text("pub const ZZ_UNTRACKED_PROBE: u8 = 1;\n", encoding="utf-8")
        scope = vr.added_names("HEAD")
        check("an UNTRACKED file's constant is in the step's scope",
              scope is not None and "ZZ_UNTRACKED_PROBE" in scope, True)
    finally:
        probe.unlink(missing_ok=True)


def every_check_is_covered():
    """The guard against 3-of-7 recurring: a new check without a test fails this file."""
    found = {n for n, f in inspect.getmembers(vr, inspect.isfunction) if n.startswith("check_")}
    check("no check is missing a test", sorted(found - set(COVERED)), [])
    check("no test names a check that is gone", sorted(set(COVERED) - found), [])


def main() -> int:
    for fn in (test_counts, test_ignored_suite, test_clippy, test_evidence, test_dataset_columns,
               test_file_checks, test_scope_is_the_step, test_untracked_files_are_in_scope, every_check_is_covered):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all verification-round checks pass'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
