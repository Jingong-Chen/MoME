# encoding: utf-8
"""
RegDB cross-modality loader using official 10-trial idx protocol.

Data layout (expected):
  <root>/regdb_raw_kaggle/RegDB/
    Visible/
    Thermal/
    idx/
      train_visible_1.txt ... train_visible_10.txt
      train_thermal_1.txt ... train_thermal_10.txt
      test_visible_1.txt  ... test_visible_10.txt
      test_thermal_1.txt  ... test_thermal_10.txt

Optional idx override:
  <root>/regdb_cross/idx/

Environment overrides:
  REGDB_CROSS_TRIAL=1..10
  REGDB_CROSS_TEST_MODE=v2t|t2v
"""

import os
import os.path as osp

from .bases import BaseImageDataset


class RegDBCross(BaseImageDataset):
    dataset_dir = "regdb_raw_kaggle/RegDB"

    def __init__(
        self,
        root="",
        trial=1,
        test_mode="v2t",
        verbose=True,
        pid_begin=0,
        **kwargs,
    ):
        super(RegDBCross, self).__init__()

        env_trial = os.environ.get("REGDB_CROSS_TRIAL", "").strip()
        env_mode = os.environ.get("REGDB_CROSS_TEST_MODE", "").strip().lower()

        if env_trial:
            trial = int(env_trial)
        if env_mode in ("v2t", "t2v"):
            test_mode = env_mode

        if trial < 1 or trial > 10:
            raise ValueError("trial must be in [1, 10], got {}".format(trial))
        if test_mode not in ("v2t", "t2v"):
            raise ValueError("test_mode must be v2t or t2v, got {}".format(test_mode))

        self.root = root
        self.pid_begin = pid_begin
        self.trial = int(trial)
        self.test_mode = test_mode

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cross_idx_dir = osp.join(root, "regdb_cross", "idx")

        if osp.isdir(self.cross_idx_dir):
            self.idx_dir = self.cross_idx_dir
        else:
            self.idx_dir = osp.join(self.dataset_dir, "idx")

        self.visible_dir = osp.join(self.dataset_dir, "Visible")
        self.thermal_dir = osp.join(self.dataset_dir, "Thermal")

        self._check_before_run()

        train_vis = self._load_idx_file("train_visible_{}.txt".format(self.trial), modality="visible")
        train_thm = self._load_idx_file("train_thermal_{}.txt".format(self.trial), modality="thermal")

        test_vis = self._load_idx_file("test_visible_{}.txt".format(self.trial), modality="visible")
        test_thm = self._load_idx_file("test_thermal_{}.txt".format(self.trial), modality="thermal")

        self.train = train_vis + train_thm
        if self.test_mode == "v2t":
            self.query = test_vis
            self.gallery = test_thm
        else:
            self.query = test_thm
            self.gallery = test_vis

        if verbose:
            print("=> RegDBCross loaded (trial={}, mode={})".format(self.trial, self.test_mode))
            print("   idx_dir: {}".format(self.idx_dir))
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.isdir(self.visible_dir):
            raise RuntimeError("'{}' is not available".format(self.visible_dir))
        if not osp.isdir(self.thermal_dir):
            raise RuntimeError("'{}' is not available".format(self.thermal_dir))
        if not osp.isdir(self.idx_dir):
            raise RuntimeError("'{}' is not available".format(self.idx_dir))

    def _resolve_image_path(self, rel_path):
        rel_path = rel_path.strip().lstrip("./")
        abs_path = osp.join(self.dataset_dir, rel_path)
        if not osp.isfile(abs_path):
            raise FileNotFoundError("Image not found: {}".format(abs_path))
        return abs_path

    def _load_idx_file(self, filename, modality):
        idx_path = osp.join(self.idx_dir, filename)
        if not osp.isfile(idx_path):
            raise FileNotFoundError("idx file not found: {}".format(idx_path))

        camid = 0 if modality == "visible" else 1
        data = []
        with open(idx_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                rel_img = parts[0]
                pid = int(parts[1])
                img_path = self._resolve_image_path(rel_img)

                data.append((img_path, self.pid_begin + pid, camid, 1))

        return data
