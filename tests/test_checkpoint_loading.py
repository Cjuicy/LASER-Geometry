import torch
from safetensors.torch import save_file


def test_load_checkpoint_state_dict_reads_safetensors(tmp_path):
    from utils.checkpoint import load_checkpoint_state_dict

    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2)}, checkpoint)

    state_dict = load_checkpoint_state_dict(str(checkpoint), map_location="cpu")

    assert set(state_dict.keys()) == {"weight"}
    torch.testing.assert_close(state_dict["weight"], torch.ones(2))


def test_load_checkpoint_state_dict_reads_wrapped_torch_checkpoint(tmp_path):
    from utils.checkpoint import load_checkpoint_state_dict

    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {"weight": torch.ones(3)}}, checkpoint)

    state_dict = load_checkpoint_state_dict(str(checkpoint), map_location="cpu")

    assert set(state_dict.keys()) == {"weight"}
    torch.testing.assert_close(state_dict["weight"], torch.ones(3))


def test_load_checkpoint_state_dict_reads_state_dict_wrapped_torch_checkpoint(tmp_path):
    from utils.checkpoint import load_checkpoint_state_dict

    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": {"weight": torch.ones(4)}}, checkpoint)

    state_dict = load_checkpoint_state_dict(str(checkpoint), map_location="cpu")

    assert set(state_dict.keys()) == {"weight"}
    torch.testing.assert_close(state_dict["weight"], torch.ones(4))
