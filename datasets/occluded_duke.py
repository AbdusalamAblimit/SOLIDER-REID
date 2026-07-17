import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class OccludedDuke(BaseImageDataset):
    """Occluded-Duke image ReID dataset built only from the raw RGB splits."""

    dataset_dir = "Occluded_Duke"
    _filename_pattern = re.compile(r"^(-?\d+)_c(\d+)_f(\d+)\.jpg$")

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(OccludedDuke, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Occluded-Duke loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for path in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.isdir(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _parse_filename(self, img_path):
        match = self._filename_pattern.fullmatch(osp.basename(img_path))
        if match is None:
            raise RuntimeError("Invalid Occluded-Duke filename: '{}'".format(img_path))
        pid, camid, _ = map(int, match.groups())
        if pid < 0:
            raise RuntimeError("Unexpected junk identity in Occluded-Duke: '{}'".format(img_path))
        if not 1 <= camid <= 8:
            raise RuntimeError("Occluded-Duke camera must be in [1, 8]: '{}'".format(img_path))
        return pid, camid

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        parsed = [(img_path, *self._parse_filename(img_path)) for img_path in img_paths]
        pid2label = {
            pid: label for label, pid in enumerate(sorted({pid for _, pid, _ in parsed}))
        }

        dataset = []
        for img_path, pid, camid in parsed:
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 1))
        return dataset
