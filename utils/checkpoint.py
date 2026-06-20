import torch


def load_checkpoint_state_dict(checkpoint_path, map_location=None):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint
