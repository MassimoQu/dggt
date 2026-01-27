import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.dataset import load_and_preprocess_images


def _sort_cav_ids(cav_ids: List[str]) -> List[str]:
    def _key(x: str):
        try:
            v = int(x)
        except Exception:
            v = 10 ** 9
        # negative ids (e.g., RSU) should be placed at the end
        return (v < 0, v)

    return sorted(cav_ids, key=_key)


def _list_timestamps(cav_path: str, camera_id: int) -> List[str]:
    ts_set = set()
    suffix = f"_camera{camera_id}.png"
    if not os.path.isdir(cav_path):
        return []
    for fname in os.listdir(cav_path):
        if not fname.endswith(suffix):
            continue
        ts = fname.split("_")[0]
        if ts:
            ts_set.add(ts)
    def _ts_key(x: str):
        try:
            return int(x)
        except Exception:
            return x
    return sorted(ts_set, key=_ts_key)


class OPV2VDataset(Dataset):
    """
    Minimal OPV2V loader for DGGT inference.

    Treats each CAV's camera{camera_id} as one view. For multi-car input,
    set views > 1 and it will pick the first N CAVs (sorted; RSU/negative ids last).
    """

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 4,
        start_idx: int = 0,
        mode: int = 2,
        views: int = 1,
        intervals: int = 2,
        camera_id: int = 0,
        max_scenes: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.start_idx = start_idx
        self.mode = mode
        self.views = views
        self.camera_id = int(camera_id)

        if mode == 3:
            self.interval = intervals
        else:
            self.interval = 1

        self.scenes = []
        self._build_index(max_scenes=max_scenes)

    def _build_index(self, max_scenes: Optional[int] = None):
        scenario_folders = sorted(
            [
                os.path.join(self.root_dir, x)
                for x in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, x))
            ]
        )
        if max_scenes is not None:
            scenario_folders = scenario_folders[: max_scenes]

        for scenario_path in scenario_folders:
            cav_ids = [
                x
                for x in os.listdir(scenario_path)
                if os.path.isdir(os.path.join(scenario_path, x))
            ]
            if not cav_ids:
                continue
            cav_ids = _sort_cav_ids(cav_ids)
            if len(cav_ids) < self.views:
                continue
            cav_ids = cav_ids[: self.views]

            # compute common timestamps across selected CAVs
            ts_lists = []
            for cav_id in cav_ids:
                cav_path = os.path.join(scenario_path, cav_id)
                ts = _list_timestamps(cav_path, self.camera_id)
                if not ts:
                    ts = []
                ts_lists.append(set(ts))

            if not ts_lists:
                continue
            common_ts = set.intersection(*ts_lists) if ts_lists else set()
            if not common_ts:
                continue
            common_ts = sorted(common_ts, key=lambda x: int(x))

            # require enough frames
            if self.mode == 3:
                required = self.start_idx + self.sequence_length * self.interval
            else:
                required = self.start_idx + self.sequence_length
            if len(common_ts) < required:
                continue

            self.scenes.append(
                {
                    "scenario_path": scenario_path,
                    "cav_ids": cav_ids,
                    "timestamps": common_ts,
                }
            )

    def __len__(self):
        return len(self.scenes)

    def _build_seq_paths(self, scene, indices: List[int]) -> List[str]:
        seq = []
        timestamps = scene["timestamps"]
        for idx in indices:
            ts = timestamps[idx]
            for cav_id in scene["cav_ids"]:
                img_path = os.path.join(
                    scene["scenario_path"],
                    cav_id,
                    f"{ts}_camera{self.camera_id}.png",
                )
                seq.append(img_path)
        return seq

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        timestamps_all = scene["timestamps"]
        if self.mode == 1:
            max_start = max(0, len(timestamps_all) - self.sequence_length)
            start = random.randint(0, max_start) if max_start > 0 else 0
            indices = [start + i for i in range(self.sequence_length)]
            intervals = [1 for _ in range(self.sequence_length - 1)]
        elif self.mode == 2:
            start = self.start_idx
            indices = [start + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
        else:  # mode == 3
            start = self.start_idx
            indices = [start + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            target_indices = [start + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

        timestamps = np.array(indices) - start
        if timestamps[-1] == 0:
            timestamps = timestamps.astype(np.float32)
        else:
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
        if self.views > 1:
            timestamps = np.repeat(timestamps, self.views)

        # images
        seq = self._build_seq_paths(scene, indices)
        images = load_and_preprocess_images(seq)

        # zero masks when sky masks are not available
        masks = torch.zeros_like(images)

        input_dict = {
            "images": images,
            "masks": masks,
            "image_paths": seq,
            "timestamps": timestamps,
            "interval": intervals,
        }

        if self.mode == 3:
            target_seq = self._build_seq_paths(scene, target_indices)
            target_images = load_and_preprocess_images(target_seq)
            target_masks = torch.zeros_like(target_images)
            input_dict.update(
                {
                    "targets": target_images,
                    "target_masks": target_masks,
                }
            )

        # no dynamic mask or depth in OPV2V by default
        if self.mode == 3:
            depth_len = len(target_seq)
        else:
            depth_len = len(seq)
        gt_depth = torch.zeros(depth_len, images.shape[2], images.shape[3])
        input_dict["gt_depth"] = gt_depth

        return input_dict
