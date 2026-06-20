import numpy as np
from scipy.spatial.transform import Rotation


def load_kitti_traj(pose_file, timestamps_file=None):
    """
    Load KITTI odometry poses from a text file of flattened 3x4 matrices.

    Returns the same tuple shape used by eval.vo_eval.load_traj:
    (traj_tum, timestamps), where traj_tum is [x, y, z, qw, qx, qy, qz].
    """
    poses = np.loadtxt(pose_file, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] not in (12, 16):
        raise ValueError(f"KITTI pose file must have 12 or 16 columns, got {poses.shape[1]}")

    if poses.shape[1] == 12:
        matrices = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
        matrices[:, :3, :4] = poses.reshape(-1, 3, 4)
    else:
        matrices = poses.reshape(-1, 4, 4)

    xyz = matrices[:, :3, 3]
    xyzw = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    traj_tum = np.column_stack([xyz, wxyz])

    if timestamps_file is None:
        timestamps = np.arange(matrices.shape[0], dtype=float)
    else:
        timestamps = np.loadtxt(timestamps_file, dtype=float)

    return traj_tum, timestamps
