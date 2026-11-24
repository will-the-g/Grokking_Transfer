"""A tiny run of the small transformer, on the CPU.

It checks that the shared code of the release still works: the model, the
data, the split, the optimizer and the evaluation. It runs a few steps of the
affine target, it writes nothing outside a temporary folder and it never
downloads a model.

    python tests/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import common  # noqa: E402


def main():
    """Five steps of the small transformer on the CPU."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    common.setup(1, device)

    model = common.make_model(common.CONFIG)
    data = common.build_task("affine", device)
    mask = common.mask_for_seed(5)
    strata, _exposure = common.split_for_mask(mask)
    optimizer = common.make_optimizer(model)
    rows = common.index_for(strata["train"][:64], device)

    before = common.evaluate(model, data, strata)
    losses = []
    for _step in range(5):
        losses.append(common.train_step(model, optimizer, data, rows))
    after = common.evaluate(model, data, strata)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smoke.pt"
        common.save_model_checkpoint(path, model, 0)
        payload = common.load_checkpoint(path)
        restored = payload["model"]

    assert after["dev_mean"] >= 0.0
    assert len(losses) == 5
    assert set(restored) == set(model.state_dict())
    print("the smoke run is fine:")
    print("  the parameters of the model : %d" % sum(p.numel() for p in model.parameters()))
    print("  the loss of the first step  : %.4f" % losses[0])
    print("  the loss of the last step   : %.4f" % losses[-1])
    print("  the dev accuracy before     : %.4f" % before["dev_mean"])
    print("  the dev accuracy after      : %.4f" % after["dev_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
