import unittest
import importlib.util
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "semantic_tapf_under_test", Path(__file__).parents[1] / "model" / "tapf.py"
)
TAPF_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TAPF_MODULE)
CleanSemanticTapfC0 = TAPF_MODULE.CleanSemanticTapfC0
SemanticSpatialRouter = TAPF_MODULE.SemanticSpatialRouter


class ExplodingPose:
    def __getitem__(self, key):
        raise AssertionError("eval must not read training-only targets")


class SemanticTapfUnitTest(unittest.TestCase):
    def make_tapf(self):
        return CleanSemanticTapfC0(
            anchor_channels=8,
            anchor_hidden=16,
            consumer_channels=12,
            router_rank=4,
            gate_release=0.5,
            gaussian_sigma=1.5,
            teacher_epochs=5,
            handoff_epochs=5,
        )

    def make_pose_batch(self, batch_size=2, all_valid=True):
        keypoints = torch.zeros(batch_size, 17, 2)
        keypoints[..., 0] = 2.0
        keypoints[..., 1] = 4.0
        scores = torch.full((batch_size, 17), 0.8)
        valid = torch.ones(batch_size, 17, dtype=torch.bool)
        semantic_valid = torch.ones(batch_size, 5, dtype=torch.bool)
        if not all_valid:
            semantic_valid[:, 2:] = False
        teacher_mask = torch.zeros(batch_size, 5, 16, 8)
        for slot in range(5):
            teacher_mask[:, slot, slot : slot + 4, :4] = 0.2 * (slot + 1)
        return {
            "keypoints": keypoints,
            "scores": scores,
            "valid": valid,
            "semantic_q_visible": torch.tensor(
                [[0.2, 0.35, 0.5, 0.65, 0.8]]
            ).expand(batch_size, -1).clone(),
            "semantic_valid": semantic_valid,
            "semantic_teacher_mask": teacher_mask,
        }

    def test_teacher_state_is_single_source_and_loss_is_frozen_mean(self):
        tapf = self.make_tapf()
        source = torch.randn(2, 8, 4, 2, requires_grad=True)
        pose_batch = self.make_pose_batch()
        state = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=1, training=True
        )
        expected_mask = torch.nn.functional.avg_pool2d(
            pose_batch["semantic_teacher_mask"], kernel_size=4, stride=4
        )
        self.assertTrue(torch.equal(state["teacher_mask"], expected_mask))
        self.assertTrue(
            torch.equal(state["teacher_presence"], pose_batch["semantic_valid"].float())
        )
        self.assertTrue(
            torch.equal(
                state["teacher_support"], pose_batch["semantic_q_visible"]
            )
        )
        self.assertTrue(torch.equal(state["mixed_mask"], expected_mask))
        self.assertTrue(
            torch.equal(state["mixed_presence"], state["teacher_presence"])
        )
        semantic_mean = torch.stack(
            [state["region_mask_loss"], state["presence_loss"], state["q_loss"]]
        ).mean()
        torch.testing.assert_close(state["semantic_loss"], semantic_mean)
        torch.testing.assert_close(
            state["pose_loss"],
            state["heatmap_loss"]
            + state["confidence_loss"]
            + semantic_mean,
        )

    def test_invalid_teacher_slot_and_all_null_student_are_exact_identity(self):
        tapf = self.make_tapf()
        source = torch.randn(2, 8, 4, 2)
        pose_batch = self.make_pose_batch(all_valid=False)
        state = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=1, training=True
        )
        self.assertTrue(
            torch.equal(
                state["consumer_mask"][:, 2:],
                torch.zeros_like(state["consumer_mask"][:, 2:]),
            )
        )
        self.assertTrue(
            torch.equal(
                state["consumer_field"][:, 2:],
                torch.zeros_like(state["consumer_field"][:, 2:]),
            )
        )

        with torch.no_grad():
            tapf.anchor.presence_head.weight.zero_()
            tapf.anchor.presence_head.bias.fill_(-1.0)
            for router in tapf.psg_bank:
                router.token_projection.weight.normal_()
                router.context_projection.weight.normal_()
                router.expert.normal_()
        eval_state = tapf.prepare(
            source,
            ExplodingPose(),
            image_hw=(8, 4),
            epoch=None,
            training=False,
        )
        self.assertTrue(
            torch.equal(
                eval_state["student_presence"],
                torch.zeros_like(eval_state["student_presence"]),
            )
        )
        tokens = torch.randn(2, 6, 12)
        routed = tapf.apply_gate(0, tokens, (3, 2), eval_state)
        self.assertTrue(torch.equal(routed, tokens))
        self.assertTrue(
            torch.equal(
                eval_state["gate_deltas"][0],
                torch.zeros_like(eval_state["gate_deltas"][0]),
            )
        )

    def test_handoff_mixes_mask_q_and_presence_with_one_fraction(self):
        tapf = self.make_tapf()
        source = torch.randn(2, 8, 4, 2)
        pose_batch = self.make_pose_batch(all_valid=False)
        state = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=6, training=True
        )
        fraction = 0.2
        expected_mask = (
            (1.0 - fraction) * state["teacher_mask"]
            + fraction * state["student_mask"]
        )
        expected_support = (
            (1.0 - fraction) * state["teacher_support"]
            + fraction * state["student_support"]
        )
        expected_presence = (
            (1.0 - fraction) * state["teacher_presence"]
            + fraction * state["student_presence"]
        )
        torch.testing.assert_close(state["mixed_mask"], expected_mask)
        torch.testing.assert_close(state["mixed_support"], expected_support)
        torch.testing.assert_close(state["mixed_presence"], expected_presence)
        torch.testing.assert_close(
            state["consumer_mask"],
            expected_mask * expected_presence[..., None, None],
        )
        torch.testing.assert_close(
            state["consumer_field"],
            expected_mask
            * expected_support[..., None, None]
            * expected_presence[..., None, None],
        )

    def test_gradient_ownership_separates_anchor_and_router(self):
        tapf = self.make_tapf()
        source = torch.randn(2, 8, 4, 2, requires_grad=True)
        state = tapf.prepare(
            source,
            self.make_pose_batch(),
            image_hw=(8, 4),
            epoch=6,
            training=True,
        )
        state["pose_loss"].backward()
        self.assertIsNone(source.grad)
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in tapf.anchor.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for router in tapf.psg_bank
                for parameter in router.parameters()
            )
        )

        tapf.zero_grad(set_to_none=True)
        tokens = torch.randn(2, 6, 12, requires_grad=True)
        with torch.no_grad():
            for router in tapf.psg_bank:
                router.expert.normal_()
        routed = tapf.apply_gate(0, tokens, (3, 2), state)
        routed.square().mean().backward()
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in tapf.anchor.parameters())
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in tapf.psg_bank[0].parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_router_zero_mask_or_zero_q_is_exact_identity(self):
        router = SemanticSpatialRouter(
            feature_channels=12, region_count=5, rank=4, release=0.5
        )
        with torch.no_grad():
            router.token_projection.weight.normal_()
            router.context_projection.weight.normal_()
            router.expert.normal_()
        tokens = torch.randn(2, 6, 12)
        mask = torch.rand(2, 5, 4, 2)
        q = torch.rand(2, 5)
        zero_mask_output, zero_mask_delta = router(
            tokens, (3, 2), torch.zeros_like(mask), q
        )
        zero_q_output, zero_q_delta = router(
            tokens, (3, 2), mask, torch.zeros_like(q)
        )
        self.assertTrue(torch.equal(zero_mask_output, tokens))
        self.assertTrue(torch.equal(zero_mask_delta, torch.zeros_like(zero_mask_delta)))
        self.assertTrue(torch.equal(zero_q_output, tokens))
        self.assertTrue(torch.equal(zero_q_delta, torch.zeros_like(zero_q_delta)))


if __name__ == "__main__":
    unittest.main()
