from collections import defaultdict
import copy
import random

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySamplerCross(Sampler):
    """
    Identity sampler with modality balancing for RegDB cross-modality training.

    For each selected PID in a mini-batch, enforce sampled instances to include
    both visible(camid=0) and thermal(camid=1) when possible.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.pid_to_all = defaultdict(list)
        self.pid_to_vis = defaultdict(list)
        self.pid_to_thm = defaultdict(list)

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            self.pid_to_all[pid].append(index)
            if int(camid) == 0:
                self.pid_to_vis[pid].append(index)
            elif int(camid) == 1:
                self.pid_to_thm[pid].append(index)

        self.pids = list(self.pid_to_all.keys())

        self.length = 0
        for pid in self.pids:
            num = len(self.pid_to_all[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def _sample_one_pid_chunk(self, pid):
        vis = self.pid_to_vis[pid]
        thm = self.pid_to_thm[pid]
        all_idx = self.pid_to_all[pid]

        n_vis = self.num_instances // 2
        n_thm = self.num_instances - n_vis

        if len(vis) > 0 and len(thm) > 0:
            if len(vis) >= n_vis:
                pick_vis = random.sample(vis, n_vis)
            else:
                pick_vis = list(np.random.choice(vis, size=n_vis, replace=True))

            if len(thm) >= n_thm:
                pick_thm = random.sample(thm, n_thm)
            else:
                pick_thm = list(np.random.choice(thm, size=n_thm, replace=True))

            chunk = pick_vis + pick_thm
        else:
            if len(all_idx) >= self.num_instances:
                chunk = random.sample(all_idx, self.num_instances)
            else:
                chunk = list(np.random.choice(all_idx, size=self.num_instances, replace=True))

        random.shuffle(chunk)
        return chunk

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_to_all[pid])
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            num_chunks = max(1, num // self.num_instances)

            for _ in range(num_chunks):
                batch_idxs_dict[pid].append(self._sample_one_pid_chunk(pid))

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
