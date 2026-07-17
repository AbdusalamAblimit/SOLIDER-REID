import unittest

import torch

from model.tapf import CleanTapfD0, PoseSpatialGate, render_pose_field


class ExplodingPose:
    def __getitem__(self, key):
        raise AssertionError("eval must not read external pose")


class CleanTapfUnitTest(unittest.TestCase):
    def make_pose_batch(self, batch_size=2):
        keypoints = torch.zeros(batch_size, 17, 2)
        keypoints[..., 0] = 2.0
        keypoints[..., 1] = 4.0
        scores = torch.full((batch_size, 17), 0.8)
        valid = torch.ones(batch_size, 17, dtype=torch.bool)
        return {"keypoints": keypoints, "scores": scores, "valid": valid}

    def make_tapf(self):
        return CleanTapfD0(
            anchor_channels=8,
            anchor_hidden=16,
            consumer_channels=12,
            psg_hidden=8,
            gaussian_sigma=1.5,
            gate_release=0.5,
            teacher_epochs=5,
            handoff_epochs=5,
        )

    def test_renderer_coordinate_reliability_and_empty_valid(self):
        keypoints = torch.tensor([[[2.0, 3.0]] * 17])
        scores = torch.ones(1, 17)
        scores[:, 0] = 1.2
        scores[:, 1] = -0.5
        valid = torch.ones(1, 17, dtype=torch.bool)
        valid[:, 2] = False
        gaussian, field, reliability = render_pose_field(
            keypoints, scores, valid, image_hw=(5, 5), field_hw=(5, 5), sigma=1.0
        )
        self.assertEqual(tuple(field.shape), (1, 17, 5, 5))
        self.assertEqual(gaussian[0, 0, 3, 2].item(), 1.0)
        self.assertEqual(reliability[0, 0].item(), 1.0)
        self.assertEqual(reliability[0, 1].item(), 0.0)
        self.assertEqual(reliability[0, 2].item(), 0.0)
        self.assertEqual(field[0, 0, 3, 2].item(), 1.0)
        self.assertTrue(torch.isfinite(field).all())

        empty = torch.zeros_like(valid)
        _, empty_field, empty_reliability = render_pose_field(
            keypoints, scores, empty, image_hw=(5, 5), field_hw=(3, 2), sigma=1.5
        )
        self.assertTrue(torch.equal(empty_field, torch.zeros_like(empty_field)))
        self.assertTrue(
            torch.equal(empty_reliability, torch.zeros_like(empty_reliability))
        )

    def test_schedule_boundaries(self):
        tapf = self.make_tapf()
        expected = {1: 0.0, 5: 0.0, 6: 0.2, 9: 0.8, 10: 1.0, 11: 1.0, 120: 1.0}
        for epoch, fraction in expected.items():
            self.assertAlmostEqual(tapf.student_fraction(epoch), fraction)

    def test_zero_field_is_exact_identity_without_bias_shortcut(self):
        gate = PoseSpatialGate(
            feature_channels=12, joint_count=17, hidden_channels=8, release=0.5
        )
        with torch.no_grad():
            gate.input_projection.weight.normal_()
            gate.output_projection.weight.normal_()
        tokens = torch.randn(2, 6, 12)
        zero_field = torch.zeros(2, 17, 4, 2)
        output, delta = gate(tokens, (3, 2), zero_field)
        self.assertTrue(torch.equal(output, tokens))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertIsNone(gate.input_projection.bias)
        self.assertIsNone(gate.output_projection.bias)
        self.assertFalse(gate.norm.affine)

    def test_route_shapes_handoff_and_gradient_isolation(self):
        tapf = self.make_tapf()
        source = torch.randn(2, 8, 4, 2, requires_grad=True)
        pose_batch = self.make_pose_batch()
        state_e1 = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=1, training=True
        )
        self.assertEqual(tuple(state_e1["student_field"].shape), (2, 17, 4, 2))
        self.assertTrue(
            torch.equal(state_e1["consumer_field"], state_e1["teacher_field"])
        )
        self.assertTrue(torch.isfinite(state_e1["pose_loss"]))
        state_e1["pose_loss"].backward()
        self.assertIsNone(source.grad)
        anchor_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in tapf.anchor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(anchor_grad, 0.0)
        self.assertTrue(
            all(parameter.grad is None for gate in tapf.psg_bank for parameter in gate.parameters())
        )

        tapf.zero_grad(set_to_none=True)
        tokens = torch.randn(2, 6, 12, requires_grad=True)
        gated = tapf.apply_gate(0, tokens, (3, 2), state_e1)
        gated = tapf.apply_gate(1, gated, (3, 2), state_e1)
        gated.square().mean().backward()
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in tapf.anchor.parameters())
        )
        output_grad = [
            float(gate.output_projection.weight.grad.abs().sum())
            for gate in tapf.psg_bank
        ]
        self.assertTrue(all(value > 0.0 for value in output_grad))
        self.assertEqual(len(state_e1["gate_deltas"]), 2)

        state_e6 = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=6, training=True
        )
        expected = 0.8 * state_e6["teacher_field"] + 0.2 * state_e6["student_field"]
        torch.testing.assert_close(state_e6["consumer_field"], expected)
        state_e10 = tapf.prepare(
            source, pose_batch, image_hw=(8, 4), epoch=10, training=True
        )
        self.assertTrue(
            torch.equal(state_e10["consumer_field"], state_e10["student_field"])
        )

    def test_eval_ignores_external_pose_and_banks_are_independent(self):
        tapf = self.make_tapf()
        source = torch.randn(1, 8, 4, 2)
        state = tapf.prepare(
            source,
            ExplodingPose(),
            image_hw=(8, 4),
            epoch=None,
            training=False,
        )
        self.assertIsNone(state["pose_loss"])
        self.assertIsNone(state["teacher_field"])
        self.assertEqual(state["student_fraction"], 1.0)
        first_ids = {id(parameter) for parameter in tapf.psg_bank[0].parameters()}
        second_ids = {id(parameter) for parameter in tapf.psg_bank[1].parameters()}
        self.assertTrue(first_ids.isdisjoint(second_ids))


if __name__ == "__main__":
    unittest.main()
