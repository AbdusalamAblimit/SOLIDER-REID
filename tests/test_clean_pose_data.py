import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from timm.data.random_erasing import RandomErasing

from datasets.paired_pose_transform import PairedPoseTransform
from datasets.pose_targets import PoseTarget, PoseTargetStore


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoseTargetStoreTest(unittest.TestCase):
    def make_artifact(self, root):
        dataset_root = root / "dataset"
        image_root = dataset_root / "bounding_box_train"
        image_root.mkdir(parents=True)
        artifact = root / "derived"
        shard_dir = artifact / "shards"
        shard_dir.mkdir(parents=True)

        relative_paths = np.asarray(
            ["bounding_box_train/a.jpg", "bounding_box_train/b.jpg"]
        )
        image_sha = []
        for name, value in (("a.jpg", 40), ("b.jpg", 80)):
            path = image_root / name
            Image.new("RGB", (4, 3), color=(value, value, value)).save(path)
            image_sha.append(sha256_file(path))
        image_sha = np.asarray(image_sha)
        sizes = np.asarray([[4, 3], [4, 3]], dtype=np.int32)
        keypoints = np.ones((2, 17, 2), dtype=np.float32)
        keypoints[0, 0, 0] = -1
        scores = np.full((2, 17), 0.75, dtype=np.float32)

        shard = shard_dir / "pose-00000.npz"
        np.savez_compressed(
            shard,
            relative_paths=relative_paths,
            image_sha256=image_sha,
            image_sizes=sizes,
            keypoints=keypoints,
            scores=scores,
        )
        records_digest = hashlib.sha256()
        for relative, rgb_sha, size, joints, confidence in zip(
            relative_paths.tolist(), image_sha.tolist(), sizes, keypoints, scores
        ):
            records_digest.update(relative.encode("utf-8"))
            records_digest.update(b"\0")
            records_digest.update(rgb_sha.encode("ascii"))
            records_digest.update(size.tobytes())
            records_digest.update(joints.tobytes())
            records_digest.update(confidence.tobytes())

        manifest = {
            "schema_version": 1,
            "joint_count": 17,
            "image_root": str(image_root),
            "sample_count": 2,
            "records_manifest_sha256": records_digest.hexdigest(),
            "shards": [
                {
                    "file": "shards/pose-00000.npz",
                    "count": 2,
                    "sha256": sha256_file(shard),
                }
            ],
        }
        manifest_path = artifact / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return artifact, image_root, sha256_file(manifest_path)

    def test_strict_load_lookup_and_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, image_root, manifest_sha = self.make_artifact(Path(temporary))
            store = PoseTargetStore(artifact, manifest_sha)
            self.assertEqual(len(store), 2)
            first = store.get(image_root / "a.jpg", verify_image_sha=True)
            self.assertEqual(first.image_size, (4, 3))
            self.assertFalse(first.valid[0].item())
            first.keypoints[1, 0] = 99
            second_read = store.get(image_root / "a.jpg")
            self.assertEqual(second_read.keypoints[1, 0].item(), 1)
            with self.assertRaises(KeyError):
                store.get(image_root.parent / "query" / "a.jpg")

    def test_manifest_and_shard_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, _, manifest_sha = self.make_artifact(Path(temporary))
            with self.assertRaises(RuntimeError):
                PoseTargetStore(artifact, "0" * 64)
            shard = artifact / "shards" / "pose-00000.npz"
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self.assertRaises(RuntimeError):
                PoseTargetStore(artifact, manifest_sha)


class PairedPoseTransformTest(unittest.TestCase):
    def make_pose(self, width, height):
        keypoints = torch.ones(17, 2, dtype=torch.float32)
        scores = torch.arange(17, dtype=torch.float32)
        valid = torch.ones(17, dtype=torch.bool)
        return PoseTarget(
            relative_path="bounding_box_train/a.jpg",
            image_sha256="a" * 64,
            image_size=(width, height),
            keypoints=keypoints,
            scores=scores,
            valid=valid,
        )

    def test_resize_flip_swap_pad_crop_and_mask(self):
        image = Image.new("RGB", (4, 2), color=(10, 20, 30))
        pose = self.make_pose(4, 2)
        pose.keypoints[0] = torch.tensor([-1.0, 1.0])
        pose.valid[0] = False
        pose.keypoints[1] = torch.tensor([0.0, 0.0])
        pose.keypoints[2] = torch.tensor([3.0, 1.0])
        transform = PairedPoseTransform(
            size_train=(4, 8),
            flip_probability=1.0,
            padding=1,
            pixel_mean=(0.0, 0.0, 0.0),
            pixel_std=(1.0, 1.0, 1.0),
            erasing_probability=0.0,
        )
        with mock.patch.object(
            T.RandomCrop, "get_params", return_value=(1, 2, 4, 8)
        ):
            rgb, augmented = transform(image, pose)
        self.assertEqual(tuple(rgb.shape), (3, 4, 8))
        self.assertTrue(augmented.flipped)
        self.assertEqual(augmented.crop_offset, (2, 1))
        self.assertEqual(augmented.image_size, (8, 4))
        torch.testing.assert_close(
            augmented.keypoints[1], torch.tensor([0.0, 2.0])
        )
        torch.testing.assert_close(
            augmented.keypoints[2], torch.tensor([6.0, 0.0])
        )
        self.assertEqual(augmented.scores[1].item(), 2.0)
        self.assertEqual(augmented.scores[2].item(), 1.0)
        self.assertFalse(augmented.valid[0].item())
        self.assertTrue(augmented.valid[1].item())
        self.assertTrue(augmented.valid[2].item())

    def test_pose_disabled_rgb_is_bit_exact_to_official_transform(self):
        pixels = np.random.RandomState(386).randint(
            0, 256, size=(73, 31, 3), dtype=np.uint8
        )
        image = Image.fromarray(pixels)
        official = T.Compose(
            [
                T.Resize((64, 32), interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(5),
                T.RandomCrop((64, 32)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                RandomErasing(
                    probability=0.5,
                    mode="pixel",
                    max_count=1,
                    device="cpu",
                ),
            ]
        )
        paired = PairedPoseTransform(
            size_train=(64, 32),
            flip_probability=0.5,
            padding=5,
            pixel_mean=(0.485, 0.456, 0.406),
            pixel_std=(0.229, 0.224, 0.225),
            erasing_probability=0.5,
        )

        for seed in range(32):
            random.seed(seed)
            torch.manual_seed(seed)
            expected = official(image)
            random.seed(seed)
            torch.manual_seed(seed)
            actual, no_pose = paired(image, None)
            self.assertIsNone(no_pose)
            self.assertTrue(torch.equal(actual, expected), msg="seed={}".format(seed))

    def test_random_erasing_changes_only_rgb(self):
        image = Image.new("RGB", (32, 64), color=(120, 100, 80))
        pose = self.make_pose(32, 64)
        common = dict(
            size_train=(64, 32),
            flip_probability=0.0,
            padding=0,
            pixel_mean=(0.0, 0.0, 0.0),
            pixel_std=(1.0, 1.0, 1.0),
        )
        clean = PairedPoseTransform(erasing_probability=0.0, **common)
        erased = PairedPoseTransform(erasing_probability=1.0, **common)
        random.seed(7)
        torch.manual_seed(7)
        clean_rgb, clean_pose = clean(image, pose)
        random.seed(7)
        torch.manual_seed(7)
        erased_rgb, erased_pose = erased(image, pose)
        self.assertFalse(torch.equal(clean_rgb, erased_rgb))
        self.assertTrue(torch.equal(clean_pose.keypoints, erased_pose.keypoints))
        self.assertTrue(torch.equal(clean_pose.scores, erased_pose.scores))
        self.assertTrue(torch.equal(clean_pose.valid, erased_pose.valid))


if __name__ == "__main__":
    unittest.main()
