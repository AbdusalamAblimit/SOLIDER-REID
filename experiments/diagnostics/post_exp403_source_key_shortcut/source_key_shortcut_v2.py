#!/usr/bin/env python3
"""Fresh v2 runner fixing only the donor slot contract from sealed-invalid v1."""

from __future__ import annotations

from typing import Dict, Sequence

import source_key_shortcut as v1


def _slot(sample: v1.Sample) -> str:
    if sample.name.startswith("q_"):
        return "q"
    return sample.name.rsplit("_", 1)[-1]


def donor_map_v2(samples: Sequence[v1.Sample]) -> Dict[str, v1.Sample]:
    by_slot = {(sample.camera, sample.pid, _slot(sample)): sample for sample in samples}
    donors: Dict[str, v1.Sample] = {}
    for sample in samples:
        donor = by_slot[(sample.camera, (sample.pid + 1) % v1.N_IDENTITIES, _slot(sample))]
        donors[sample.name] = donor
    return donors


v1.donor_map = donor_map_v2


if __name__ == "__main__":
    raise SystemExit(v1.main())
