import os

import numpy as np
import torch

import envs.add_env as add_env
import envs.deepmimic_env as deepmimic_env
import util.torch_util as torch_util


class HeightmapADDEnv(add_env.ADDEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        heightmap_config = env_config.get("heightmap", {})
        self._enable_heightmap_obs = env_config.get("heightmap_obs", heightmap_config.get("enable", True))
        self._heightmap_num_rows = int(env_config.get("heightmap_num_rows", heightmap_config.get("num_rows", 15)))
        self._heightmap_num_cols = int(env_config.get("heightmap_num_cols", heightmap_config.get("num_cols", 15)))
        self._heightmap_x_range = env_config.get("heightmap_x_range", heightmap_config.get("x_range", [-1.5, 1.5]))
        self._heightmap_y_range = env_config.get("heightmap_y_range", heightmap_config.get("y_range", [-1.0, 1.0]))
        self._heightmap_ground_height = float(env_config.get("heightmap_ground_height", heightmap_config.get("ground_height", 0.0)))
        self._heightmap_relative_to_root = bool(env_config.get("heightmap_relative_to_root", heightmap_config.get("relative_to_root", True)))
        self._heightmap_clip = env_config.get("heightmap_clip", heightmap_config.get("clip", None))
        self._heightmap_file = env_config.get("heightmap_file", heightmap_config.get("file", None))
        self._heightmap_origin = env_config.get("heightmap_origin", heightmap_config.get("origin", [0.0, 0.0]))
        self._heightmap_resolution = float(env_config.get("heightmap_resolution", heightmap_config.get("resolution", 0.05)))
        self._heightmap_data = None

        assert self._heightmap_num_rows > 0
        assert self._heightmap_num_cols > 0
        assert len(self._heightmap_x_range) == 2
        assert len(self._heightmap_y_range) == 2

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        self._build_heightmap_tensors()
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        if self._enable_heightmap_obs:
            heightmap_obs = self._compute_heightmap_obs(env_ids)
            obs = torch.cat([obs, heightmap_obs], dim=-1)

        return obs

    def _update_reward(self):
        # ADDEnv inherits AMPEnv, whose task reward is intentionally empty.
        # For heightmap motion tracking we need the DeepMimic tracking reward
        # so ADDAgent's task_reward_weight actually constrains root/pose.
        deepmimic_env.DeepMimicEnv._update_reward(self)
        return

    def _build_heightmap_tensors(self):
        xs = torch.linspace(self._heightmap_x_range[0], self._heightmap_x_range[1],
                            self._heightmap_num_cols, device=self._device, dtype=torch.float32)
        ys = torch.linspace(self._heightmap_y_range[0], self._heightmap_y_range[1],
                            self._heightmap_num_rows, device=self._device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs)
        offsets = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros_like(xx).reshape(-1)], dim=-1)
        self._heightmap_sample_offsets = offsets
        self._heightmap_origin_tensor = torch.tensor(self._heightmap_origin, device=self._device, dtype=torch.float32)

        if self._heightmap_clip is None:
            self._heightmap_clip_tensor = None
        else:
            assert len(self._heightmap_clip) == 2
            self._heightmap_clip_tensor = torch.tensor(self._heightmap_clip, device=self._device, dtype=torch.float32)

        if self._heightmap_file is not None and self._heightmap_file != "":
            heightmap_path = self._heightmap_file
            if not os.path.isabs(heightmap_path):
                heightmap_path = os.path.abspath(heightmap_path)

            heightmap = np.load(heightmap_path)
            assert len(heightmap.shape) == 2, "heightmap_file must point to a 2D .npy array"
            self._heightmap_data = torch.tensor(heightmap, device=self._device, dtype=torch.float32)

        return

    def _compute_heightmap_obs(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]

        n = root_pos.shape[0]
        num_samples = self._heightmap_sample_offsets.shape[0]

        heading_rot = torch_util.calc_heading_quat(root_rot)
        heading_rot = heading_rot.unsqueeze(1).expand(-1, num_samples, -1)
        offsets = self._heightmap_sample_offsets.unsqueeze(0).expand(n, -1, -1)
        world_offsets = torch_util.quat_rotate(heading_rot.reshape(-1, 4),
                                               offsets.reshape(-1, 3)).reshape(n, num_samples, 3)
        sample_xy = root_pos[:, None, 0:2] + world_offsets[..., 0:2]
        heights = self._sample_heightmap(sample_xy)

        if self._heightmap_relative_to_root:
            heights = heights - root_pos[:, 2:3]

        if self._heightmap_clip_tensor is not None:
            heights = torch.clamp(heights, self._heightmap_clip_tensor[0], self._heightmap_clip_tensor[1])

        return heights.reshape(n, -1)

    def _sample_heightmap(self, sample_xy):
        if self._heightmap_data is None:
            heights = torch.full(sample_xy.shape[:-1], self._heightmap_ground_height,
                                 device=self._device, dtype=sample_xy.dtype)
        else:
            heights = self._sample_heightmap_bilinear(sample_xy)

        return heights

    def _sample_heightmap_bilinear(self, sample_xy):
        heightmap = self._heightmap_data
        h = heightmap.shape[0]
        w = heightmap.shape[1]

        grid_xy = (sample_xy - self._heightmap_origin_tensor) / self._heightmap_resolution
        gx = torch.clamp(grid_xy[..., 0], 0.0, float(w - 1))
        gy = torch.clamp(grid_xy[..., 1], 0.0, float(h - 1))

        x0 = torch.floor(gx).long()
        y0 = torch.floor(gy).long()
        x1 = torch.clamp(x0 + 1, max=w - 1)
        y1 = torch.clamp(y0 + 1, max=h - 1)

        wx = gx - x0.to(gx.dtype)
        wy = gy - y0.to(gy.dtype)

        flat_heightmap = heightmap.reshape(-1)
        h00 = flat_heightmap[y0 * w + x0]
        h01 = flat_heightmap[y0 * w + x1]
        h10 = flat_heightmap[y1 * w + x0]
        h11 = flat_heightmap[y1 * w + x1]

        h0 = h00 * (1.0 - wx) + h01 * wx
        h1 = h10 * (1.0 - wx) + h11 * wx
        heights = h0 * (1.0 - wy) + h1 * wy
        return heights
