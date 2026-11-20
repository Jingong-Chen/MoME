#!/usr/bin/env python3
"""Every entry point starts.

Byte-compiling a file proves it parses, not that its imports resolve. The scripts under scripts/
were written at a different depth in the tree they came from, so several of them resolved the
repository root one directory too high and died on "No module named config" before reaching
main(). Nothing in the test suite noticed, because nothing ran them.

This runs each entry point with --help from the repository root and fails on the errors that mean
the file cannot start at all. A script that gets as far as looking for a dataset or an artifact
has started, so a missing checkpoint under logs/ or a missing dataset under data/ passes here.

  python tests/check_entrypoints.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FATAL = ("ModuleNotFoundError", "ImportError", "SyntaxError", "IndentationError",
         "NameError", "AttributeError: module", "unsupported format character")


def entry_points():
    out = []
    for d in (ROOT, os.path.join(ROOT, "scripts")):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith(".py") and os.path.isfile(p):
                src = open(p, encoding="utf-8").read()
                if "__main__" in src or "argparse" in src:
                    out.append(os.path.relpath(p, ROOT))
    return out


def main():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)            # the scripts have to find the repository themselves
    env["MPLBACKEND"] = "Agg"
    bad = []
    names = entry_points()
    for rel in names:
        r = subprocess.run([sys.executable, rel, "--help"], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        blob = r.stdout + r.stderr
        hit = next((k for k in FATAL if k in blob), None)
        if hit:
            bad.append((rel, [ln for ln in blob.splitlines() if hit in ln][-1].strip()))
            print("FAIL %s" % rel)
        else:
            print("ok   %s" % rel)
    print("\n%d entry points, %d cannot start" % (len(names), len(bad)))
    for rel, line in bad:
        print("  %s: %s" % (rel, line))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
