#!/usr/bin/env python3
import argparse
import os
import shutil
from collections import defaultdict


def find_visible_dir(root_dir):
    candidates = ["Visible", "visible", "RGB", "rgb", "Vis", "vis"]
    for base, dirs, _ in os.walk(root_dir):
        for c in candidates:
            if c in dirs:
                return os.path.join(base, c)
    return None


def gather_pid_images(visible_dir):
    pid_to_paths = defaultdict(list)
    for d in sorted(os.listdir(visible_dir)):
        p = os.path.join(visible_dir, d)
        if not os.path.isdir(p):
            continue
        try:
            pid = int(d)
        except ValueError:
            continue
        for fn in sorted(os.listdir(p)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                pid_to_paths[pid].append(os.path.join(p, fn))
    return pid_to_paths


def main():
    parser = argparse.ArgumentParser(description="Convert RegDB visible subset to TransReID layout")
    parser.add_argument(
        "--regdb_raw",
        default=os.path.join(os.environ.get("MOME_DATA", "./data"), "regdb_raw"),
        type=str,
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.environ.get("MOME_DATA", "./data"), "regdb"),
        type=str,
    )
    args = parser.parse_args()

    visible_dir = find_visible_dir(args.regdb_raw)
    if visible_dir is None:
        raise RuntimeError(
            "Could not find visible/RGB directory under {}".format(args.regdb_raw)
        )
    print("Visible dir:", visible_dir)

    pid_to_paths = gather_pid_images(visible_dir)
    all_pids = sorted(pid_to_paths.keys())
    if len(all_pids) < 4:
        raise RuntimeError("Too few IDs found in {}".format(visible_dir))

    n_total = sum(len(v) for v in pid_to_paths.values())
    print("Found IDs:", len(all_pids), "images:", n_total)

    # Fixed split: first half for training, second half for test.
    split = len(all_pids) // 2
    train_pids = all_pids[:split]
    test_pids = all_pids[split:]
    print("Train IDs:", len(train_pids), "Test IDs:", len(test_pids))

    train_dir = os.path.join(args.output, "bounding_box_train")
    gallery_dir = os.path.join(args.output, "bounding_box_test")
    query_dir = os.path.join(args.output, "query")
    for d in (train_dir, gallery_dir, query_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    cnt_train = cnt_gallery = cnt_query = 0
    # Train uses all visible images from train IDs.
    for pid in train_pids:
        for src in pid_to_paths[pid]:
            dst = os.path.join(train_dir, "{:04d}_c1s1_{:06d}_regdb.jpg".format(pid, cnt_train))
            shutil.copy2(src, dst)
            cnt_train += 1

    # Test IDs: split each PID images into gallery/query halves.
    for pid in test_pids:
        imgs = pid_to_paths[pid]
        half = max(1, len(imgs) // 2)
        gallery_imgs = imgs[:half]
        query_imgs = imgs[half:]
        if not query_imgs:
            query_imgs = gallery_imgs[-1:]
            gallery_imgs = gallery_imgs[:-1] if len(gallery_imgs) > 1 else gallery_imgs

        for src in gallery_imgs:
            dst = os.path.join(gallery_dir, "{:04d}_c1s1_{:06d}_regdb.jpg".format(pid, cnt_gallery))
            shutil.copy2(src, dst)
            cnt_gallery += 1
        for src in query_imgs:
            dst = os.path.join(query_dir, "{:04d}_c2s1_{:06d}_regdb.jpg".format(pid, cnt_query))
            shutil.copy2(src, dst)
            cnt_query += 1

    print("Output:", args.output)
    print("  Train:", cnt_train)
    print("  Gallery:", cnt_gallery)
    print("  Query:", cnt_query)


if __name__ == "__main__":
    main()
