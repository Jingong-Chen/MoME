#!/usr/bin/env python3
"""Copy official RegDB idx files into data/regdb_cross/idx and validate counts."""

import argparse
import os
import shutil


def count_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=os.path.join(os.environ.get("MOME_DATA", "./data"), "regdb_raw_kaggle/RegDB/idx"),
        type=str,
    )
    parser.add_argument(
        "--dst",
        default=os.path.join(os.environ.get("MOME_DATA", "./data"), "regdb_cross/idx"),
        type=str,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise FileNotFoundError("source idx dir not found: {}".format(args.src))

    os.makedirs(args.dst, exist_ok=True)

    expected = []
    for t in range(1, 11):
        expected.extend(
            [
                "train_visible_{}.txt".format(t),
                "train_thermal_{}.txt".format(t),
                "test_visible_{}.txt".format(t),
                "test_thermal_{}.txt".format(t),
            ]
        )

    print("Source:", args.src)
    print("Target:", args.dst)

    for name in expected:
        src_f = os.path.join(args.src, name)
        dst_f = os.path.join(args.dst, name)
        if not os.path.isfile(src_f):
            raise FileNotFoundError("missing official idx file: {}".format(src_f))
        shutil.copy2(src_f, dst_f)

    print("Copied {} files.".format(len(expected)))

    for name in expected:
        fpath = os.path.join(args.dst, name)
        n = count_lines(fpath)
        print("  {:24s} {}".format(name, n))


if __name__ == "__main__":
    main()
