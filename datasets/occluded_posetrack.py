# encoding: utf-8
"""Occluded-PoseTrack-ReID (KPR, ECCV2024) 数据读取。

crop 已按 Market 格式组织：bounding_box_train / query / bounding_box_test，
文件名形如 <pid>_c<camid>_<frameid>.jpg（camid 是视频派生编号）。

评测口径：KPR 报告用的是 mot_inter_intra_video，即每个 query 与全部 gallery 比对、
不做任何同相机/同视频去除。utils/metrics.py 的 eval_func 会删掉 (同 pid 且 同 camid)
的 gallery 样本，所以这里让 query 全部 camid=0、gallery 全部 camid=1（两者不相交），
去除条件 (同 pid & 同 camid) 恒为假，等价于与全部 gallery 比对。
注意：必须保持 SIE_CAMERA 关闭（测试集视频与训练集不重叠，把视频号当相机嵌入会泄漏/失配）。
"""
import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class OccludedPoseTrack(BaseImageDataset):
    dataset_dir = 'occluded_posetrack_reid'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OccludedPoseTrack, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True, fixed_camid=None)
        query = self._process_dir(self.query_dir, relabel=False, fixed_camid=0)
        gallery = self._process_dir(self.gallery_dir, relabel=False, fixed_camid=1)

        if verbose:
            print("=> Occluded-PoseTrack-ReID loaded")
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

    def _process_dir(self, dir_path, relabel=False, fixed_camid=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(pattern.search(osp.basename(img_path)).group(1))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            base = osp.basename(img_path)
            m = pattern.search(base)
            pid = int(m.group(1))
            camid = int(m.group(2)) if fixed_camid is None else fixed_camid
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
