# encoding: utf-8
"""Dataset definition for Occluded-ReID."""

import os
import os.path as osp
import re

from .bases import BaseImageDataset


class OccludedREID(BaseImageDataset):
    """Occluded-ReID dataset loader.

    Expected layout:

        occluded_reid/
            query/      -> occluded images, nested by identity
            gallery/    -> full-body images, nested by identity

    This dataset has no official training split, so ``train`` is empty.
    Query images are assigned camera 0 and gallery images camera 1 so
    positive matches are not filtered out as same-camera duplicates.
    """

    default_dataset_dir = "occluded_reid"

    def __init__(self, root="", dataset_dir=None, verbose=True,
                 pid_begin=0, **kwargs):
        super().__init__()

        if dataset_dir is None:
            dataset_dir = self.default_dataset_dir
        if osp.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = osp.join(root, dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")

        self._check_before_run()

        self.pid_begin = pid_begin
        train = []
        query = self._process_dir(self.query_dir, camid=0)
        gallery = self._process_dir(self.gallery_dir, camid=1)

        if verbose:
            print("=> Occluded-ReID loaded")
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

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, camid):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_paths = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                if osp.splitext(name)[1].lower() in exts and not name.startswith('.'):
                    img_paths.append(osp.join(root, name))
        img_paths = sorted(img_paths)

        pattern = re.compile(r'([-\d]+)_\d+')
        dataset = []
        for img_path in img_paths:
            parent = osp.basename(osp.dirname(img_path))
            match = pattern.search(osp.basename(img_path))

            if parent.isdigit():
                pid = int(parent)
            elif match is not None:
                pid = int(match.group(1))
            else:
                raise RuntimeError(
                    "Image '{}' does not contain a valid pid".format(img_path))

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
