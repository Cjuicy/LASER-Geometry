import sys
import types

import torch


def _njit_stub(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


numba_stub = types.ModuleType("numba")
numba_stub.njit = _njit_stub
sys.modules.setdefault("numba", numba_stub)

from loop_closure.loop_closure import LoopClosureEngine


def test_loop_closure_prefers_registration_fields_when_present():
    prediction = {
        "local_points": torch.zeros(2, 1, 1, 3),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "registration_local_points": torch.ones(2, 1, 1, 3),
        "registration_camera_poses": torch.eye(4).repeat(2, 1, 1) * 2,
    }

    point_map, camera_poses = LoopClosureEngine._registration_fields(prediction)

    assert point_map is prediction["registration_local_points"]
    assert camera_poses is prediction["registration_camera_poses"]


def test_loop_closure_registration_fields_fall_back_to_raw_predictions():
    prediction = {
        "local_points": torch.zeros(2, 1, 1, 3),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
    }

    point_map, camera_poses = LoopClosureEngine._registration_fields(prediction)

    assert point_map is prediction["local_points"]
    assert camera_poses is prediction["camera_poses"]
