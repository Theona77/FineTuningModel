"""Import BTC (third_party/BTC-ISMIR19) unmodified, with shims for current libraries.

The 2019 code uses np.int / np.float (removed in numpy 2), yaml.load without a Loader
(rejected by PyYAML 6), and its checkpoints hold numpy scalars (rejected by torch >= 2.6's
weights_only default). Import BTC things from here rather than from the submodule directly.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

BTC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "BTC-ISMIR19"
CKPT_LARGE = BTC_DIR / "test" / "btc_model_large_voca.pt"
CKPT_MAJMIN = BTC_DIR / "test" / "btc_model.pt"

for _name, _type in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)
if not getattr(yaml.load, "_btc_shim", False):
    _yaml_load = yaml.load
    yaml.load = functools.wraps(_yaml_load)(
        lambda stream, Loader=yaml.FullLoader: _yaml_load(stream, Loader=Loader))
    yaml.load._btc_shim = True

if not (BTC_DIR / "btc_model.py").exists():
    raise ImportError(f"BTC not found at {BTC_DIR}; run: git submodule update --init")
sys.path.append(str(BTC_DIR))     # after our own modules: BTC also has a train.py
from btc_model import BTC_model, bi_directional_self_attention_layers  # noqa: E402,F401
from utils.hparams import HParams  # noqa: E402,F401
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord  # noqa: E402,F401


def load_checkpoint(path: Path = CKPT_LARGE) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def btc_config():
    return HParams.load(str(BTC_DIR / "run_config.yaml"))
