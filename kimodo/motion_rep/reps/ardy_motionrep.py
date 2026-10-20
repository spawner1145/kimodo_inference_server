# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ardy motion representation for autoregressive inference."""

import einops
import torch

from ...geometry import cont6d_to_matrix, matrix_to_cont6d
from ...skeleton.kinematics import fk, joints_from_global_rots
from ...skeleton.transforms import global_rots_to_local_rots
from ...tools import ensure_batched, to_numpy
from ..conditioning import ConditionWriters
from ..feature_utils import (
    RotateFeatures,
    compute_heading_angle,
    compute_vel_xyz,
)
from ..feet import foot_detect_from_pos_and_vel
from .base import FeatureLayout, MotionRepBase


class ArdyMotionRep(MotionRepBase):
    """Global root / global joint representation used by Ardy inference.

    Feature layout:
    - ``root_pos``: root position ``[x, y, z]``.
    - ``global_root_heading``: root heading as ``[cos(theta), sin(theta)]``.
    - ``local_joints_positions``: non-root joints in root-local coordinates.
    - ``global_rot_data``: global joint rotations in 6D representation.
    - ``velocities``: global joint velocities.
    - ``foot_contacts``: four foot contact channels.
    """

    def __init__(
        self,
        skeleton,
        fps,
        stats_path: str | None = None,
        stats=None,
        name: str | None = None,
        **kwargs,
    ):
        # `stats`, `name`, and **kwargs let ArdyMotionRep be built straight from the
        # training config via Hydra instantiate(), which passes a skeleton, a Stats
        # object (stats=...), a `name`, and possibly extra keys.
        assert skeleton.root_idx == 0, "ArdyMotionRep assumes the skeleton root index is 0."
        # "ardy" is the canonical rep name (the on-disk feature folders were renamed from the
        # legacy "<skeleton>_dual_root_global_joints" — pure rename, identical features)
        self.name = name if name is not None else f"{skeleton.name}_ardy"
        # Stats object (with a .folder) -> reuse ardy's single-folder sliced stats loader.
        if stats_path is None and stats is not None:
            stats_path = getattr(stats, "folder", None)
        nbjoints = skeleton.nbjoints

        layout = FeatureLayout(
            size_dict={
                "root_pos": torch.Size([3]),
                "global_root_heading": torch.Size([2]),
                "local_joints_positions": torch.Size([nbjoints - 1, 3]),  # removed the pelvis joint
                "global_rot_data": torch.Size([nbjoints, 6]),
                "velocities": torch.Size([nbjoints, 3]),
                "foot_contacts": torch.Size([4]),
            },
            last_root_feature="global_root_heading",
            local_root_size_dict={
                "local_root_rot_vel": torch.Size([1]),
                "local_root_vel": torch.Size([2]),
                "global_root_y": torch.Size([1]),
            },
        )
        super().__init__(skeleton, fps, stats_path, layout=layout)

    @ensure_batched(local_joint_rots=5, root_positions=3, lengths=1)
    def __call__(
        self,
        local_joint_rots: torch.Tensor,
        root_positions: torch.Tensor,
        to_normalize: bool,
        to_canonicalize: bool = False,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert local rotations and root positions into Ardy features."""
        device = local_joint_rots.device
        if lengths is None:
            assert local_joint_rots.shape[0] == 1, "If lengths is not provided, the input should not be batched."
            lengths = torch.tensor([local_joint_rots.shape[1]], device=device)

        global_joint_rots, global_positions, local_joints_positions_origin_is_pelvis = fk(
            local_joint_rots,
            root_positions,
            self.skeleton,
        )

        root_heading_angle = compute_heading_angle(global_positions, self.skeleton)
        global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)

        ground_offset = torch.zeros_like(root_positions)
        ground_offset[..., 1] = root_positions[..., 1]
        # remove the pelvis joint (root_idx == 0) and align onto the ground
        local_joints_positions = local_joints_positions_origin_is_pelvis[:, :, 1:] + ground_offset[:, :, None]

        velocities = compute_vel_xyz(global_positions, self.fps, lengths=lengths)
        foot_contacts = foot_detect_from_pos_and_vel(global_positions, velocities, self.skeleton, 0.15, 0.10)
        global_rot_data = matrix_to_cont6d(global_joint_rots)

        features, _ = einops.pack(
            [
                root_positions,
                global_root_heading,
                local_joints_positions,
                global_rot_data,
                velocities,
                foot_contacts,
            ],
            "batch time *",
        )

        assert features.shape[-1] == self.motion_rep_dim

        if to_canonicalize:
            features = self.canonicalize(features, normalized=False)

        if to_normalize:
            features = self.normalize(features)
        return features

    @ensure_batched(features=3, angle=1)
    def rotate(self, features: torch.Tensor, angle: torch.Tensor):
        """Rotate root/joint positional and rotational features by heading."""
        # assume it is not normalized
        bs = features.shape[0]
        device = features.device
        [
            root_pos,
            global_root_heading,
            local_joints_positions,
            global_rot_data,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=device)
        if len(angle.shape) == 0:
            angle = angle.repeat(bs)

        RF = RotateFeatures(angle)
        new_features, _ = einops.pack(
            [
                RF.rotate_positions(root_pos),
                RF.rotate_2d_positions(global_root_heading),
                RF.rotate_positions(local_joints_positions),
                RF.rotate_6d_rotations(global_rot_data),
                RF.rotate_positions(velocities),
                foot_contacts,
            ],
            "batch time *",
        )
        return new_features

    @ensure_batched(features=3, translation_2d=2)
    def translate_2d(self, features: torch.Tensor, translation_2d: torch.Tensor) -> torch.Tensor:
        """Translate root planar position by ``(dx, dz)``."""
        bs = features.shape[0]
        if len(translation_2d.shape) == 1:
            translation_2d = translation_2d.repeat(bs, 1)

        new_features = features.clone()
        new_root_pos = new_features[:, :, self.slice_dict["root_pos"]]
        new_root_pos[:, :, 0] += translation_2d[:, [0]]
        new_root_pos[:, :, 2] += translation_2d[:, [1]]
        return new_features

    @ensure_batched(features=3)
    def inverse(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        posed_joints_from="rotations",
        return_numpy: bool = False,
        return_all: bool = False,
        need_local_rots: bool = True,
    ) -> dict:
        """Decode Ardy features into motion tensors.

        ``return_all`` adds an ``all_positions`` dict with the joints decoded from the position and
        rotation blocks (``{"positions": ..., "rotations": ...}``). ``need_local_rots=False`` skips
        the global->local conversion (``local_rot_mats`` is then ``None`` in the output) — the
        training losses only consume positions / global rotations.
        """
        assert posed_joints_from in ["rotations", "positions"], "posed_joints_from should be rotations or positions"

        if is_normalized:
            features = self.unnormalize(features)

        [
            root_positions,
            global_root_heading,
            local_joints_positions,
            global_rot_data,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        global_rot_mats = cont6d_to_matrix(global_rot_data)
        # skeletons with a baked rest pose re-bake it inside fk, so their position path still
        # needs the local rotations; everywhere else positions come straight from the globals
        has_baked_rest = getattr(self.skeleton, "rest_local_rots", None) is not None
        local_rot_mats = (
            global_rots_to_local_rots(global_rot_mats, self.skeleton) if (need_local_rots or has_baked_rest) else None
        )

        posed_joints_from_rots = posed_joints_from_pos = None
        # root-local pose (root at the origin), returned by the FK so the position loss can use it
        # directly instead of re-subtracting the root
        posed_joints_norootpos = None
        if return_all or posed_joints_from == "rotations":
            if has_baked_rest:
                _, posed_joints_from_rots, posed_joints_norootpos = fk(local_rot_mats, root_positions, self.skeleton)
            else:
                posed_joints_from_rots, posed_joints_norootpos = joints_from_global_rots(
                    global_rot_mats, root_positions, self.skeleton, return_norootpos=True
                )
        if return_all or posed_joints_from == "positions":
            # the encoder drops the pelvis row; restore it at the root height (the other rows
            # already carry the absolute height via the encoder's ground offset)
            dummy_root = torch.zeros_like(local_joints_positions[:, :, [0]])
            dummy_root[..., 1] = root_positions[..., None, 1]
            posed_joints_from_pos = torch.cat([dummy_root, local_joints_positions], dim=2)
            posed_joints_from_pos[..., 0] += root_positions[..., None, 0]
            posed_joints_from_pos[..., 2] += root_positions[..., None, 2]
        posed_joints = posed_joints_from_rots if posed_joints_from == "rotations" else posed_joints_from_pos

        output_tensor_dict = {
            "local_rot_mats": local_rot_mats,
            "global_rot_mats": global_rot_mats,
            "posed_joints": posed_joints,
            "posed_joints_norootpos": posed_joints_norootpos,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions,
            "foot_contacts": foot_contacts > 0.5,
            "global_root_heading": global_root_heading,
        }
        if return_all:
            output_tensor_dict["all_positions"] = {
                "positions": posed_joints_from_pos,
                "rotations": posed_joints_from_rots,
            }
        if return_numpy:
            return to_numpy(output_tensor_dict)
        return output_tensor_dict

    def create_conditions(
        self,
        index_dict: dict,
        data_dict: dict,
        length: int,
        to_normalize: bool,
        device: str,
    ):
        """Build sparse conditioning tensors for the ardy representation.

        Returns:
            ``(observed_motion, motion_mask)`` where ``motion_mask`` is **float** and unmasked
            cells of ``observed_motion`` are zeroed. This deliberately differs from
            ``KimodoMotionRep.create_conditions`` (bool mask, unmasked cells left as normalized
            zeros) — callers are contract-specific.
        """
        # create empty features and mask to be filled in
        observed_motion = torch.zeros(length, self.motion_rep_dim, device=device)
        motion_mask = torch.zeros(length, self.motion_rep_dim, dtype=bool, device=device)

        writers = ConditionWriters(
            observed_motion,
            motion_mask,
            self.slice_dict,
            index_dict,
            data_dict,
            device,
            joints_per_block=lambda fname: self.size_dict[fname][0],
        )
        _gather, _write_frames, _write_joints = writers.gather, writers.write_frames, writers.write_joints

        root_2d_name = "root_2d" if index_dict.get("root_2d") else "smooth_root_2d"
        if (gathered := _gather(root_2d_name)) is not None:
            indices, root_pos_2d = gathered
            _write_frames("root_pos", indices, root_pos_2d, components=[0, 2])

        if (gathered := _gather("root_y_pos")) is not None:
            indices, root_y_pos = gathered
            _write_frames("root_pos", indices, root_y_pos, components=[1])

        if (gathered := _gather("global_root_heading")) is not None:
            indices, global_root_heading = gathered
            _write_frames("global_root_heading", indices, global_root_heading)

        if (gathered := _gather("global_joints_rots")) is not None:
            indices_lst, global_joints_rots = gathered
            _write_joints("global_rot_data", indices_lst, matrix_to_cont6d(global_joints_rots))

        if (gathered := _gather("global_joints_positions")) is not None:
            indices_lst, global_joints_positions = gathered
            time_indices = indices_lst[:, 0].contiguous()
            root_mask = motion_mask[time_indices, self.slice_dict["root_pos"]]
            if not root_mask[:, [0, 2]].all():
                raise ValueError("For constraining global positions, root 2D should also be constrained.")

            # joint positions are stored relative to the ground-projected root: (x, z) relative, y absolute.
            # Same as kimodo: the reference is the root 2D already written above (every pose
            # constraint also constrains the root 2D on its frames), so hands/feet-only constraints work.
            local_reference = observed_motion[time_indices, self.slice_dict["root_pos"]].clone()
            local_reference[..., 1] = 0.0
            local_joints_positions = global_joints_positions - local_reference

            # the local_joints_positions block excludes the root joint, so drop it and shift joint indices
            non_root = indices_lst[:, 1] != self.skeleton.root_idx
            pair_indices = indices_lst[non_root].clone()
            pair_indices[:, 1] -= 1
            _write_joints("local_joints_positions", pair_indices, local_joints_positions[non_root])

        motion_mask = motion_mask.float()
        if to_normalize:
            observed_motion = self.normalize(observed_motion) * motion_mask
        return observed_motion, motion_mask
