# encoding: utf-8
"""Occluded-Duke 数据读取（标准遮挡 ReID 基准，作 MULTIHYP 的跨数据集验证）。

标准 Market 格式：bounding_box_train / query / bounding_box_test，
文件名 <pid>_c<camid>_f<frame>.jpg（camid 真实、c1..c8）。
评测口径：标准单 query 协议，eval_func 去除 (同 pid 且 同 camid) 的 gallery（与 Market 一致）。
这与 Occ-PoseTrack 的"不去除"不同——Duke 用真实 camid、走标准去除。SIE_CAMERA 保持关闭。
"""
import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class OccludedDuke(BaseImageDataset):
    dataset_dir = 'Occluded_Duke'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OccludedDuke, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

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
        for d in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available".format(d))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(pattern.search(osp.basename(img_path)).group(1))
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            m = pattern.search(osp.basename(img_path))
            pid, camid = int(m.group(1)), int(m.group(2))
            if pid == -1:
                continue
            camid -= 1  # camid 转 0 基（仅用于同相机去除，SIE 关闭）
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
