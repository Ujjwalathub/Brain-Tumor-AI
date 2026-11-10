"""CPU vs GPU forward divergence — controlled reproduction."""

import sys

sys.path.insert(0, "src")

import torch

from utils import load_config
from models import create_model

CKPT = "models/best_epoch019_macro-recall0.9929.pth"


def fwd(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    config = load_config("configs/efficientnet_b0.yaml")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # 1. Random-init CPU baseline (no checkpoint)
    m = create_model(config)
    print("random init  CPU ", fwd(m, x).round(3))

    # 2. Checkpoint weights on CPU, as stored (81 tensors channels_last)
    m = create_model(config)
    m.load_state_dict(sd)
    print("ckpt (cf)    CPU ", fwd(m, x).round(3))

    # 3. Checkpoint weights forced contiguous-format on CPU
    m = create_model(config)
    m.load_state_dict({k: v.contiguous(memory_format=torch.contiguous_format) for k, v in sd.items()})
    print("ckpt (cfmt)  CPU ", fwd(m, x).round(3))

    # 4. Same contiguous weights on GPU, no channels_last
    if torch.cuda.is_available():
        m = create_model(config)
        m.load_state_dict({k: v.contiguous(memory_format=torch.contiguous_format) for k, v in sd.items()})
        m = m.cuda()
        print("ckpt (cfmt)  GPU ", fwd(m, x.cuda()).round(3))

    # 5. GPU + channels_last (the training/eval path)
    if torch.cuda.is_available():
        m = create_model(config)
        m.load_state_dict(sd)
        m = m.cuda().to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model_fp16 = m(x.cuda()).float().cpu().numpy()
        print("ckpt (cf+fp16)GPU", out.round(3))


if __name__ == "__main__":
    main()
