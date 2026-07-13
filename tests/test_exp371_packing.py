from experiments.exp371_casd.intervention_utils import descriptor_gain_retention


def test_retention_uses_paired_gain_not_absolute_map():
    assert abs(descriptor_gain_retention(0.5972, 0.5900, 0.5990) - 0.8) < 1e-9


def test_retention_rejects_nonpositive_teacher_gain():
    try:
        descriptor_gain_retention(0.6, 0.6, 0.6)
    except ValueError as exc:
        assert 'full descriptor must beat global' in str(exc)
    else:
        raise AssertionError('nonpositive teacher gain must invalidate Gate D')
