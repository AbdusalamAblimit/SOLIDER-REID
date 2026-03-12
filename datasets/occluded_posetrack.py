# encoding: utf-8
"""Dataset definition for Occluded-PoseTrack-ReID.

Same directory structure as Occluded-DukeMTMC:
    occluded_posetrack_reid/
        bounding_box_train/
        bounding_box_test/
        query/
        train.list
        gallery.list
        query.list
"""

import re
import os.path as osp

from .bases import BaseImageDataset


class OccludedPoseTrack(BaseImageDataset):
    """Occluded-PoseTrack-ReID dataset loader.

    Derived from PoseTrack21 using KPR's sampling annotations.
    Format is identical to Occluded-DukeMTMC.
    """

    dataset_dir = "occluded_posetrack_reid"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.train_list = osp.join(self.dataset_dir, "train.list")
        self.query_list = osp.join(self.dataset_dir, "query.list")
        self.gallery_list = osp.join(self.dataset_dir, "gallery.list")

        self._check_before_run()

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, self.train_list, relabel=True)
        query = self._process_dir(self.query_dir, self.query_list, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_list, relabel=False)

        if verbose:
            print("=> Occluded-PoseTrack-ReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (self.num_train_pids,
         self.num_train_imgs,
         self.num_train_cams,
         self.num_train_vids) = self.get_imagedata_info(self.train)
        (self.num_query_pids,
         self.num_query_imgs,
         self.num_query_cams,
         self.num_query_vids) = self.get_imagedata_info(self.query)
        (self.num_gallery_pids,
         self.num_gallery_imgs,
         self.num_gallery_cams,
         self.num_gallery_vids) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self) -> None:
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.exists(self.query_list):
            raise RuntimeError("'{}' is not available".format(self.query_list))
        if not osp.exists(self.gallery_list):
            raise RuntimeError("'{}' is not available".format(self.gallery_list))

    def _process_dir(self, dir_path: str, list_path: str, relabel: bool = False):
        with open(list_path, "r") as list_file:
            img_names = [line.strip() for line in list_file if line.strip()]

        img_paths = []
        for img_name in img_names:
            img_path = osp.join(dir_path, img_name)
            if not osp.exists(img_path):
                raise RuntimeError(
                    "Image '{}' listed in '{}' is missing".format(img_name, list_path))
            img_paths.append(img_path)

        pattern = re.compile(r"([-\d]+)_c(\d+)")

        pid_container = set()
        for img_path in sorted(img_paths):
            match = pattern.search(osp.basename(img_path))
            if match is None:
                raise RuntimeError(
                    "Image name '{}' does not follow expected format".format(img_path))
            pid, _ = map(int, match.groups())
            if pid == -1:
                continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            if pid < 0:
                raise RuntimeError(
                    "Image '{}' has invalid pid '{}'".format(img_path, pid))
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
