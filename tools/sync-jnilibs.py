#!/usr/bin/env python3
"""Keep the shipped `.so` honest: detect when it is older than the Rust it was built from, and rebuild.

    python whoop-rs/tools/sync-jnilibs.py --check   # exit 1 if the shipped libraries are stale
    python whoop-rs/tools/sync-jnilibs.py --sync    # rebuild both ABIs, regenerate bindings, stamp
    python whoop-rs/tools/sync-jnilibs.py --ensure  # rebuild only when stale; what Gradle calls

The host DLL is safe already: `app/build.gradle.kts` runs `cargo build -p whoop-ffi` before every test
task, so cargo tracks its freshness. The libraries that reach a PHONE are different - they are committed
artifacts under `jniLibs/`, and nothing rebuilt or checked them. On 2026-07-31 they were 17 hours and 13
library source files behind `crates/`, while the JVM suite passed 2,527 green against a freshly built host
library. Green tests, stale device. That is the one failure mode of the three that nothing caught: a
MISSING library fails loudly, a missing FIXTURE self-skips (which is why `skipped = 0` is checked), and a
STALE library does neither.

The fingerprint covers `crates/*/src/**.rs` plus every `Cargo.toml` and `Cargo.lock`. That is a SUPERSET of
what `whoop-ffi` links, so a change to a crate the app never loads still asks for a rebuild. Over-rebuilding
is the safe direction; under-detecting is what shipped stale code.

Nothing here is silent. A missing toolchain is an error, never a skip: `onlyIf { Cargo.toml.exists() }` in
the Gradle task is exactly how a detached worktree ended up testing against whatever library was lying
around.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WHOOP = ROOT.parent
# The app tree the artifacts land in. `--app-src` retargets it: several worktrees of the app share this
# one checkout, and writing into the wrong one leaves another tree holding artifacts it did not build.
# The worktrees were consolidated into `noop` on 2026-08-14; the old default pointed at a deleted tree
# and `:app:syncRustJniLibs` failed on every clean checkout until 2026-09-10.
APP_SRC = WHOOP / "noop" / "android" / "app" / "src"
JNILIBS = APP_SRC / "main" / "jniLibs"
BINDINGS = APP_SRC / "main" / "java" / "uniffi" / "whoop_ffi"
STAMP = JNILIBS / ".fingerprint"
ABIS = ("arm64-v8a", "x86_64")
LIB = "libwhoop_ffi.so"


def retarget(app_src: Path) -> None:
    global APP_SRC, JNILIBS, BINDINGS, STAMP
    APP_SRC = app_src.resolve()
    JNILIBS = APP_SRC / "main" / "jniLibs"
    BINDINGS = APP_SRC / "main" / "java" / "uniffi" / "whoop_ffi"
    STAMP = JNILIBS / ".fingerprint"


def fingerprint() -> str:
    """A content hash of every input IN THE REPOSITORY that can change what the shipped library does.

    Verified a superset of what `whoop-ffi` links: no crate has a `build.rs`, no `include_str!`/
    `include_bytes!` and no non-`.rs` file under `src/`, and the crate pulls only `whoop-protocol`,
    `physio-algo` and `uniffi` (whose version is in the lock file). The build ENVIRONMENT - the rustc
    and NDK versions - is outside it, so a toolchain bump reads as no change.
    """
    h = hashlib.sha256()
    files = sorted((ROOT / "crates").rglob("*.rs"))
    files = [p for p in files if "/src/" in p.as_posix()]
    files += sorted((ROOT / "crates").rglob("Cargo.toml"))
    for p in [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]:
        if p.exists():
            files.append(p)
    for p in sorted(set(files)):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def stamped() -> str | None:
    return STAMP.read_text(encoding="utf-8").strip() if STAMP.exists() else None


def missing_libs() -> list[str]:
    return [abi for abi in ABIS if not (JNILIBS / abi / LIB).exists()]


def require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"error: `{tool}` is not on PATH. Refusing to leave the shipped library stale.")


def run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"error: {cmd[0]} failed with {r.returncode}")


def sync() -> int:
    """The four steps an FFI change needs, in one command, then stamp what they were built from."""
    require("cargo")
    require("cargo-ndk")
    want = fingerprint()

    run(["cargo", "build", "--release", "-p", "whoop-ffi"], ROOT)

    lib = next((ROOT / "target" / "release" / n for n in ("whoop_ffi.dll", "libwhoop_ffi.so", "libwhoop_ffi.dylib")
                if (ROOT / "target" / "release" / n).exists()), None)
    if lib is None:
        raise SystemExit("error: no host library after `cargo build`; nothing to generate bindings from.")
    BINDINGS.mkdir(parents=True, exist_ok=True)
    # The generator is a BIN inside whoop-ffi behind the `cli` feature, not a package of its own, and
    # uniffi writes `<out-dir>/uniffi/whoop_ffi/whoop_ffi.kt`, so the out-dir is the java source root.
    run(["cargo", "run", "--release", "-p", "whoop-ffi", "--bin", "uniffi-bindgen", "--features", "cli", "--",
         "generate", "--library", str(lib), "--language", "kotlin",
         "--out-dir", str(BINDINGS.parent.parent)], ROOT)

    # The `-t` list and the list [missing_libs] verifies are one constant, or the check guards an ABI
    # the build never produced.
    targets = [a for abi in ABIS for a in ("-t", abi)]
    run(["cargo", "ndk", *targets, "-o", str(JNILIBS), "build", "--release", "-p", "whoop-ffi"], ROOT)

    if gone := missing_libs():
        raise SystemExit(f"error: cargo ndk did not produce {gone}; not stamping a build that did not happen.")

    STAMP.write_text(want + "\n", encoding="utf-8")
    print(f"\nsynced and stamped {want[:12]} for {', '.join(sorted(ABIS))}")
    return 0


def check() -> int:
    want, have = fingerprint(), stamped()
    print(f"whoop-rs crates fingerprint {want[:12]}   jniLibs stamp {(have or 'ABSENT')[:12]}")
    if gone := missing_libs():
        print(f"  STALE  the shipped library is missing for {gone}")
    elif have is None:
        print("  STALE  no fingerprint beside the shipped libraries, so what they were built from is unknown")
    elif have != want:
        print("  STALE  the shipped libraries were built from different Rust than the tree holds")
    else:
        print("  ok     the shipped libraries match the tree they were built from")
        return 0
    print("\n  run: python whoop-rs/tools/sync-jnilibs.py --sync")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness and exit 1 when stale")
    ap.add_argument("--sync", action="store_true", help="rebuild both ABIs, regenerate bindings, stamp")
    ap.add_argument("--ensure", action="store_true",
                    help="rebuild ONLY when stale; the build's entry point, cheap when nothing moved")
    ap.add_argument("--app-src", type=Path, default=None,
                    help="the app's src/ to write bindings + jniLibs into (default: the noop checkout)")
    args = ap.parse_args()
    if args.app_src is not None:
        retarget(args.app_src)
    print(f"app tree {APP_SRC}")
    if not JNILIBS.is_dir():
        raise SystemExit(f"error: {JNILIBS} does not exist; wrong checkout layout.")
    if args.sync:
        return sync()
    if args.ensure:
        # Hashing the tree costs milliseconds; a rebuild costs minutes. Only pay it when it is owed.
        return 0 if check() == 0 else sync()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
