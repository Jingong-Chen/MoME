#!/usr/bin/env python3
"""
Run existing TransReID scripts with runtime RegDBCross registration,
without modifying existing code.

Example:
  python scripts/run_with_regdb_cross.py \
    --script train.py --trial 1 --test_mode v2t -- \
    --config_file configs/RegDB/vit_transreid_stride_cross.yml \
    MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/regdb_cross_standard/trial_1
"""

import argparse
import os
import runpy
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, type=str)
    parser.add_argument("--trial", default=1, type=int)
    parser.add_argument("--test_mode", default="v2t", choices=["v2t", "t2v"])
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    from utils.regdb_cross_patch import patch_regdb_cross_factories, set_regdb_cross_env

    script_path = args.script
    if not os.path.isfile(script_path):
        script_path = os.path.join(os.getcwd(), args.script)
    if not os.path.isfile(script_path):
        raise FileNotFoundError("script not found: {}".format(args.script))

    # Scripts with local __factory should use dedicated *_cross scripts.
    bad = {"train_mome_v3.py", "test_mome_v3.py", "eval_partition.py"}
    if os.path.basename(script_path) in bad:
        raise RuntimeError(
            "{} defines local __factory; run dedicated cross script instead.".format(
                os.path.basename(script_path)
            )
        )

    set_regdb_cross_env(trial=args.trial, test_mode=args.test_mode)
    patch_regdb_cross_factories()

    passthrough = list(args.script_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    sys.argv = [script_path] + passthrough
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
