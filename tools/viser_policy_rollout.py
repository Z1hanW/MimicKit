#!/usr/bin/env python3
"""Viser-powered policy rollout viewer for MimicKit tracking policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MIMICKIT_DIR = ROOT_DIR / "mimickit"
VISER_SRC = ROOT_DIR.parents[1] / "vis" / "utils" / "viser" / "src"

# Keep this before importing torch-facing MimicKit modules. MimicKit's engine
# builder imports isaacgym first, which avoids Isaac Gym's torch import warning.
sys.path.insert(0, str(MIMICKIT_DIR))
if VISER_SRC.exists():
    sys.path.insert(0, str(VISER_SRC))

# Must be first among MimicKit imports. engine_builder imports isaacgym before
# any module below pulls in torch.
import engines.engine_builder as _isaacgym_import_guard  # noqa: F401
import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.arg_parser as mk_arg_parser
import util.mp_util as mp_util

import torch

try:
    import viser
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import viser. Install it in the active environment, for example:\n"
        "  python -m pip install -e /home/ubuntu/FAR/SGM/vis/utils/viser"
    ) from exc


DEFAULT_ARG_FILE = "args/add_smpl_generalsit_heightmap_track_curriculum_args.txt"
DEFAULT_ACTIVE_OUT_DIR = "output/add_sgm_8gpu_track_curriculum_20260707_012908"
DONE_NAMES = {
    base_env.DoneFlags.NULL.value: "NULL",
    base_env.DoneFlags.FAIL.value: "FAIL",
    base_env.DoneFlags.SUCC.value: "SUCC",
    base_env.DoneFlags.TIME.value: "TIME",
}


@dataclass
class RolloutBundle:
    motion_index: int
    motion_file: str
    start_time: float
    timestep: float
    policy_body_pos: np.ndarray
    ref_body_pos: np.ndarray
    policy_root_pos: np.ndarray
    ref_root_pos: np.ndarray
    contact_mask: np.ndarray
    rewards: np.ndarray
    done_flags: np.ndarray
    motion_times: np.ndarray
    root_pos_err: np.ndarray
    body_pos_err: np.ndarray
    body_pos_max_err: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.policy_body_pos.shape[0])

    @property
    def duration(self) -> float:
        return max(0, self.num_frames - 1) * self.timestep


class RolloutRunner:
    def __init__(
        self,
        arg_file: Path,
        model_file: Path,
        device: str,
        num_envs: int,
        master_port: int,
    ) -> None:
        os.chdir(str(ROOT_DIR))
        mp_util.init(0, 1, device, master_port)

        self.device = device
        self.args = load_mimickit_args(arg_file)
        self.env = build_env(self.args, num_envs, device)
        self.agent = build_agent(self.args, self.env, device)
        self.agent.load(str(model_file))
        self.agent.eval()
        self.agent.set_mode(base_agent.AgentMode.TEST)

        self.motion_files = list(self.env._motion_lib._motion_files)
        self.motion_lengths = self.env._motion_lib.get_motion_lengths().detach().cpu().numpy()
        self.motion_names = [Path(path).name for path in self.motion_files]
        self.body_names = list(self.env._kin_char_model.get_body_names())
        self.parent_indices = np.asarray(self.env._kin_char_model._parent_indices, dtype=np.int64)
        self.edges = np.asarray(
            [(parent, child) for child, parent in enumerate(self.parent_indices) if parent >= 0],
            dtype=np.int64,
        )
        self.char_id = self.env._get_char_id()
        self.env_ids = torch.tensor([0], device=self.env._device, dtype=torch.long)
        self.timestep = float(self.env._engine.get_timestep())

    def generate(
        self,
        motion_index: int,
        start_time: float,
        max_steps: int,
        stop_on_done: bool,
        contact_force_threshold: float,
    ) -> RolloutBundle:
        motion_index = int(np.clip(motion_index, 0, len(self.motion_files) - 1))
        motion_len = float(self.motion_lengths[motion_index])
        start_time = float(np.clip(start_time, 0.0, max(0.0, motion_len - self.timestep)))

        self.agent.eval()
        self.agent.set_mode(base_agent.AgentMode.TEST)

        with torch.no_grad():
            obs, info = self._reset_fixed_motion(motion_index, start_time)

            policy_body_pos: List[np.ndarray] = []
            ref_body_pos: List[np.ndarray] = []
            policy_root_pos: List[np.ndarray] = []
            ref_root_pos: List[np.ndarray] = []
            contact_mask: List[np.ndarray] = []
            rewards: List[float] = []
            done_flags: List[int] = []
            motion_times: List[float] = []
            root_pos_err: List[float] = []
            body_pos_err: List[float] = []
            body_pos_max_err: List[float] = []

            def capture() -> int:
                body = self.env._engine.get_body_pos(self.char_id)[0].detach().cpu().numpy().copy()
                ref_body = self.env._ref_body_pos[0].detach().cpu().numpy().copy()
                root = self.env._engine.get_root_pos(self.char_id)[0].detach().cpu().numpy().copy()
                ref_root = self.env._ref_root_pos[0].detach().cpu().numpy().copy()
                contacts = self.env._engine.get_ground_contact_forces(self.char_id)[0]
                contacts = contacts.detach().cpu().numpy()
                curr_done = int(self.env._done_buf[0].detach().cpu().item())

                body_dist = np.linalg.norm(body - ref_body, axis=-1)
                policy_body_pos.append(body)
                ref_body_pos.append(ref_body)
                policy_root_pos.append(root)
                ref_root_pos.append(ref_root)
                contact_mask.append(np.linalg.norm(contacts, axis=-1) > contact_force_threshold)
                rewards.append(float(self.env._reward_buf[0].detach().cpu().item()))
                done_flags.append(curr_done)
                motion_times.append(float(self.env._get_motion_times()[0].detach().cpu().item()))
                root_pos_err.append(float(np.linalg.norm(root - ref_root)))
                body_pos_err.append(float(np.mean(body_dist)))
                body_pos_max_err.append(float(np.max(body_dist)))
                return curr_done

            capture()
            for _ in range(max_steps):
                action, _ = self.agent._decide_action(obs, info)
                obs, _, done, info = self.agent._step_env(action)
                curr_done = capture()
                if stop_on_done and curr_done != base_env.DoneFlags.NULL.value:
                    break

        return RolloutBundle(
            motion_index=motion_index,
            motion_file=self.motion_files[motion_index],
            start_time=start_time,
            timestep=self.timestep,
            policy_body_pos=np.asarray(policy_body_pos, dtype=np.float32),
            ref_body_pos=np.asarray(ref_body_pos, dtype=np.float32),
            policy_root_pos=np.asarray(policy_root_pos, dtype=np.float32),
            ref_root_pos=np.asarray(ref_root_pos, dtype=np.float32),
            contact_mask=np.asarray(contact_mask, dtype=np.bool_),
            rewards=np.asarray(rewards, dtype=np.float32),
            done_flags=np.asarray(done_flags, dtype=np.int32),
            motion_times=np.asarray(motion_times, dtype=np.float32),
            root_pos_err=np.asarray(root_pos_err, dtype=np.float32),
            body_pos_err=np.asarray(body_pos_err, dtype=np.float32),
            body_pos_max_err=np.asarray(body_pos_max_err, dtype=np.float32),
        )

    def _reset_fixed_motion(self, motion_index: int, start_time: float):
        self.env.reset(self.env_ids)
        motion_ids = torch.full((1,), motion_index, device=self.env._device, dtype=torch.long)
        motion_times = torch.full((1,), start_time, device=self.env._device, dtype=torch.float32)

        self.env._motion_ids[self.env_ids] = motion_ids
        self.env._motion_time_offsets[self.env_ids] = motion_times - self.env._time_buf[self.env_ids]

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = (
            self.env._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )
        body_pos, body_rot = self.env._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        dof_pos = self.env._motion_lib.joint_rot_to_dof(joint_rot)

        self.env._ref_root_pos[self.env_ids] = root_pos
        self.env._ref_root_rot[self.env_ids] = root_rot
        self.env._ref_root_vel[self.env_ids] = root_vel
        self.env._ref_root_ang_vel[self.env_ids] = root_ang_vel
        self.env._ref_joint_rot[self.env_ids] = joint_rot
        self.env._ref_dof_vel[self.env_ids] = dof_vel
        self.env._ref_body_pos[self.env_ids] = body_pos
        self.env._ref_body_rot[self.env_ids] = body_rot
        self.env._ref_dof_pos[self.env_ids] = dof_pos

        self.env._engine.set_root_pos(self.env_ids, self.char_id, root_pos)
        self.env._engine.set_root_rot(self.env_ids, self.char_id, root_rot)
        self.env._engine.set_root_vel(self.env_ids, self.char_id, root_vel)
        self.env._engine.set_root_ang_vel(self.env_ids, self.char_id, root_ang_vel)
        self.env._engine.set_dof_pos(self.env_ids, self.char_id, dof_pos)
        self.env._engine.set_dof_vel(self.env_ids, self.char_id, dof_vel)
        self.env._engine.set_body_pos(self.env_ids, self.char_id, body_pos)
        self.env._engine.set_body_rot(self.env_ids, self.char_id, body_rot)
        self.env._engine.set_body_vel(self.env_ids, self.char_id, 0.0)
        self.env._engine.set_body_ang_vel(self.env_ids, self.char_id, 0.0)

        # Root/dof setters only queue an Isaac Gym reset. Drain that queue now so
        # the first policy action is evaluated from the same z-up state we show.
        if hasattr(self.env._engine, "_update_reset_objs"):
            self.env._engine._update_reset_objs()
        if hasattr(self.env._engine, "_refresh_sim_tensors"):
            self.env._engine._refresh_sim_tensors()

        # Rigid-body tensors may remain stale until a sim tick after a reset.
        # Keep the immediate observation/display consistent with the target FK.
        self.env._engine.set_body_pos(self.env_ids, self.char_id, body_pos)
        self.env._engine.set_body_rot(self.env_ids, self.char_id, body_rot)
        self.env._engine.set_body_vel(self.env_ids, self.char_id, 0.0)
        self.env._engine.set_body_ang_vel(self.env_ids, self.char_id, 0.0)

        if hasattr(self.env, "_reset_disc_hist"):
            self.env._reset_disc_hist(self.env_ids)

        self.env._update_observations(self.env_ids)
        self.env._update_info(self.env_ids)
        return self.env._obs_buf, self.env._info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a viser policy rollout viewer.")
    parser.add_argument("--arg-file", type=Path, default=Path(DEFAULT_ARG_FILE))
    parser.add_argument(
        "--model-file",
        type=str,
        default="latest",
        help="Checkpoint file, output dir, or 'latest' for the active SGM run.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=6951)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--motion-query", default="", help="Initial motion basename/index substring.")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--continue-after-done", action="store_true")
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--search-limit", type=int, default=80)
    parser.add_argument("--policy-color", default="70,155,255")
    parser.add_argument("--ref-color", default="255,190,70")
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_mimickit_args(arg_file: Path) -> mk_arg_parser.ArgParser:
    arg_file = resolve_repo_path(arg_file)
    args = mk_arg_parser.ArgParser()
    if not args.load_file(str(arg_file)):
        raise RuntimeError("Failed to load arg file: {}".format(arg_file))
    return args


def build_env(args: mk_arg_parser.ArgParser, num_envs: int, device: str):
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    return env_builder.build_env(
        env_file=env_file,
        engine_file=engine_file,
        num_envs=num_envs,
        device=device,
        visualize=False,
        record_video=False,
    )


def build_agent(args: mk_arg_parser.ArgParser, env, device: str):
    agent_file = args.parse_string("agent_config")
    return agent_builder.build_agent(agent_file, env, device)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_model_file(model_arg: str) -> Path:
    if model_arg == "latest":
        return latest_checkpoint(ROOT_DIR / DEFAULT_ACTIVE_OUT_DIR)

    path = Path(model_arg)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if path.is_dir():
        return latest_checkpoint(path)
    if not path.exists():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    return path


def latest_checkpoint(out_dir: Path) -> Path:
    candidates: List[Path] = []
    int_dir = out_dir / "int_models"
    if int_dir.exists():
        now = time.time()
        candidates.extend(
            path
            for path in int_dir.glob("model_*.pt")
            if path.is_file() and now - path.stat().st_mtime > 5.0
        )
    model_file = out_dir / "model.pt"
    if model_file.exists() and time.time() - model_file.stat().st_mtime > 5.0:
        candidates.append(model_file)
    if not candidates:
        raise FileNotFoundError("No stable checkpoint found under {}".format(out_dir))
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_rgb(text: str) -> Tuple[int, int, int]:
    vals = tuple(int(x) for x in text.split(","))
    if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
        raise ValueError("Expected RGB text like 70,155,255")
    return vals  # type: ignore[return-value]


def make_skeleton_segments(body_pos: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([body_pos[edges[:, 0]], body_pos[edges[:, 1]]], axis=1).astype(np.float32)


def make_trajectory_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


def camera_from_points(points: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    flat = points.reshape(-1, 3)
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo) * 0.5), 1.5)
    position = center + np.array([radius * 1.7, radius * -1.6, radius * 1.1], dtype=np.float32)
    return tuple(position.tolist()), tuple(center.tolist())


def grid_from_points(points: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    flat = points.reshape(-1, 3)
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    center = (lo + hi) * 0.5
    width = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 4.0) * 1.4
    return width, (float(center[0]), float(center[1]), 0.0)


def shifted_ref(bundle: RolloutBundle, offset_x: float) -> Tuple[np.ndarray, np.ndarray]:
    shift = np.array([offset_x, 0.0, 0.0], dtype=np.float32)
    return bundle.ref_body_pos + shift, bundle.ref_root_pos + shift


def bundle_markdown(bundle: RolloutBundle) -> str:
    final_done = DONE_NAMES.get(int(bundle.done_flags[-1]), str(int(bundle.done_flags[-1])))
    policy_body_flat = bundle.policy_body_pos.reshape(-1, 3)
    policy_body0 = bundle.policy_body_pos[0].reshape(-1, 3)
    ref_body_flat = bundle.ref_body_pos.reshape(-1, 3)
    ref_body0 = bundle.ref_body_pos[0].reshape(-1, 3)
    return "\n".join(
        [
            "Motion: `{}`".format(Path(bundle.motion_file).name),
            "Index: `{}`".format(bundle.motion_index),
            "Coordinate: `z-up (+z vertical)`, ground: `z=0`",
            "Start: `{:.2f}s`, frames: `{}`, duration: `{:.2f}s`".format(
                bundle.start_time, bundle.num_frames, bundle.duration
            ),
            "Ref root z: `{:.3f}..{:.3f}`, ref body z: `{:.3f}..{:.3f}`".format(
                float(np.min(bundle.ref_root_pos[:, 2])),
                float(np.max(bundle.ref_root_pos[:, 2])),
                float(np.min(ref_body_flat[:, 2])),
                float(np.max(ref_body_flat[:, 2])),
            ),
            "Policy root z: `{:.3f}..{:.3f}`, policy body z: `{:.3f}..{:.3f}`".format(
                float(np.min(bundle.policy_root_pos[:, 2])),
                float(np.max(bundle.policy_root_pos[:, 2])),
                float(np.min(policy_body_flat[:, 2])),
                float(np.max(policy_body_flat[:, 2])),
            ),
            "Frame 0 root z policy/ref: `{:.3f}`/`{:.3f}`, body min z policy/ref: `{:.3f}`/`{:.3f}`".format(
                float(bundle.policy_root_pos[0, 2]),
                float(bundle.ref_root_pos[0, 2]),
                float(np.min(policy_body0[:, 2])),
                float(np.min(ref_body0[:, 2])),
            ),
            "Final done: `{}`".format(final_done),
            "Mean body err: `{:.4f}`, max body err: `{:.4f}`".format(
                float(np.mean(bundle.body_pos_err)), float(np.max(bundle.body_pos_max_err))
            ),
            "Mean root err: `{:.4f}`".format(float(np.mean(bundle.root_pos_err))),
        ]
    )


def frame_markdown(bundle: RolloutBundle, frame: int) -> str:
    done_name = DONE_NAMES.get(int(bundle.done_flags[frame]), str(int(bundle.done_flags[frame])))
    return (
        "Frame `{}` / `{}` | t=`{:.3f}s` | reward=`{:.4f}` | "
        "root_err=`{:.4f}` | body_err=`{:.4f}` | max_body_err=`{:.4f}` | done=`{}`"
    ).format(
        frame,
        bundle.num_frames - 1,
        float(bundle.motion_times[frame]),
        float(bundle.rewards[frame]),
        float(bundle.root_pos_err[frame]),
        float(bundle.body_pos_err[frame]),
        float(bundle.body_pos_max_err[frame]),
        done_name,
    )


def resolve_initial_motion(runner: RolloutRunner, motion_index: int, motion_query: str) -> int:
    if motion_query.strip():
        return resolve_motion_text(runner, motion_query)
    return int(np.clip(motion_index, 0, len(runner.motion_files) - 1))


def resolve_motion_text(runner: RolloutRunner, text: str) -> int:
    query = text.strip()
    if not query:
        return 0
    prefix = query.split(":", 1)[0].strip()
    if prefix.isdigit():
        idx = int(prefix)
        if 0 <= idx < len(runner.motion_files):
            return idx
    q = query.lower()
    for i, name in enumerate(runner.motion_names):
        if q == name.lower() or q == Path(name).stem.lower():
            return i
    for i, path in enumerate(runner.motion_files):
        if q in path.lower() or q in runner.motion_names[i].lower():
            return i
    raise ValueError("Could not resolve motion: {}".format(text))


def motion_label(runner: RolloutRunner, idx: int) -> str:
    return "{}: {}".format(idx, runner.motion_names[idx])


def motion_matches(runner: RolloutRunner, query: str, limit: int) -> List[int]:
    q = query.strip().lower()
    if not q:
        return list(range(min(limit, len(runner.motion_files))))
    matches = [
        i
        for i, path in enumerate(runner.motion_files)
        if q in path.lower() or q in runner.motion_names[i].lower()
    ]
    return matches[:limit]


def serve_viewer(args: argparse.Namespace, runner: RolloutRunner, initial_bundle: RolloutBundle) -> None:
    policy_color = parse_rgb(args.policy_color)
    ref_color = parse_rgb(args.ref_color)
    contact_color = (255, 70, 70)
    current_bundle = initial_bundle
    current_motion_index = initial_bundle.motion_index
    state_lock = threading.RLock()
    state: Dict[str, object] = {"frame": 0, "last_tick": time.monotonic(), "loading": False}

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True
    server.scene.configure_default_lights()
    if args.share:
        server.request_share_url()

    ref_body, ref_root = shifted_ref(current_bundle, 0.0)
    all_points = np.concatenate([current_bundle.policy_body_pos, ref_body], axis=1)
    cam_pos, cam_look_at = camera_from_points(all_points)
    server.initial_camera.position = cam_pos
    server.initial_camera.look_at = cam_look_at

    grid_width, grid_pos = grid_from_points(all_points)
    grid_handle = server.scene.add_grid("/ground_z0_grid", width=grid_width, height=grid_width, position=grid_pos, plane="xy")
    z_axis_points = np.asarray(
        [[[grid_pos[0], grid_pos[1], 0.0], [grid_pos[0], grid_pos[1], 2.0]]],
        dtype=np.float32,
    )
    z_axis = server.scene.add_line_segments(
        "/debug/z_up_axis",
        points=z_axis_points,
        colors=(40, 255, 120),
        line_width=8.0,
    )
    z_axis_label = server.scene.add_label(
        "/debug/z_up_axis_label",
        "+Z up",
        position=(grid_pos[0], grid_pos[1], 2.08),
        font_screen_scale=1.5,
        anchor="bottom-center",
    )
    ground_label = server.scene.add_label(
        "/debug/ground_label",
        "ground z=0",
        position=(grid_pos[0], grid_pos[1], 0.02),
        font_screen_scale=1.1,
        anchor="top-center",
    )

    policy_skel = server.scene.add_line_segments(
        "/policy/skeleton",
        points=make_skeleton_segments(current_bundle.policy_body_pos[0], runner.edges),
        colors=policy_color,
        line_width=args.line_width,
    )
    ref_skel = server.scene.add_line_segments(
        "/reference/skeleton",
        points=make_skeleton_segments(ref_body[0], runner.edges),
        colors=ref_color,
        line_width=args.line_width,
    )
    policy_joints = server.scene.add_point_cloud(
        "/policy/joints",
        points=current_bundle.policy_body_pos[0],
        colors=policy_color,
        point_size=0.035,
        point_shape="circle",
    )
    ref_joints = server.scene.add_point_cloud(
        "/reference/joints",
        points=ref_body[0],
        colors=ref_color,
        point_size=0.025,
        point_shape="circle",
    )
    policy_traj = server.scene.add_line_segments(
        "/policy/root_trajectory",
        points=make_trajectory_segments(current_bundle.policy_root_pos),
        colors=policy_color,
        line_width=2.0,
    )
    ref_traj = server.scene.add_line_segments(
        "/reference/root_trajectory",
        points=make_trajectory_segments(ref_root),
        colors=ref_color,
        line_width=2.0,
    )
    policy_root = server.scene.add_icosphere(
        "/policy/root_marker",
        radius=0.045,
        color=policy_color,
        subdivisions=2,
        position=tuple(current_bundle.policy_root_pos[0].tolist()),
    )
    ref_root_marker = server.scene.add_icosphere(
        "/reference/root_marker",
        radius=0.035,
        color=ref_color,
        subdivisions=2,
        position=tuple(ref_root[0].tolist()),
    )
    contact_points0 = current_bundle.policy_body_pos[0][current_bundle.contact_mask[0]]
    contact_points = server.scene.add_point_cloud(
        "/policy/ground_contacts",
        points=contact_points0.astype(np.float32).reshape(-1, 3),
        colors=contact_color,
        point_size=0.075,
        point_shape="circle",
        visible=contact_points0.shape[0] > 0,
    )

    def set_client_camera(bundle: RolloutBundle) -> None:
        ref_body_new, _ = shifted_ref(bundle, float(ref_offset.value))
        points = np.concatenate([bundle.policy_body_pos, ref_body_new], axis=1)
        position, look_at = camera_from_points(points)
        server.initial_camera.position = position
        server.initial_camera.look_at = look_at
        for client in server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = look_at

    def set_frame(frame: int) -> None:
        nonlocal current_bundle
        with state_lock:
            if bool(state["loading"]):
                return
            bundle = current_bundle
            frame = int(np.clip(frame, 0, bundle.num_frames - 1))
            state["frame"] = frame
            ref_body_curr, ref_root_curr = shifted_ref(bundle, float(ref_offset.value))
            contacts = bundle.policy_body_pos[frame][bundle.contact_mask[frame]]
            with server.atomic():
                policy_skel.points = make_skeleton_segments(bundle.policy_body_pos[frame], runner.edges)
                ref_skel.points = make_skeleton_segments(ref_body_curr[frame], runner.edges)
                policy_joints.points = bundle.policy_body_pos[frame]
                ref_joints.points = ref_body_curr[frame]
                policy_root.position = tuple(bundle.policy_root_pos[frame].tolist())
                ref_root_marker.position = tuple(ref_root_curr[frame].tolist())
                contact_points.points = contacts.astype(np.float32).reshape(-1, 3)
                contact_points.visible = bool(show_contacts.value and contacts.shape[0] > 0)
                if frame_slider.value != frame:
                    frame_slider.value = frame
                frame_status.content = frame_markdown(bundle, frame)
            server.flush()

    def refresh_ref_offset() -> None:
        ref_body_curr, ref_root_curr = shifted_ref(current_bundle, float(ref_offset.value))
        frame = int(state["frame"])
        with server.atomic():
            ref_skel.points = make_skeleton_segments(ref_body_curr[frame], runner.edges)
            ref_joints.points = ref_body_curr[frame]
            ref_traj.points = make_trajectory_segments(ref_root_curr)
            ref_root_marker.position = tuple(ref_root_curr[frame].tolist())
        server.flush()

    def set_visual_toggles() -> None:
        policy_skel.visible = bool(show_policy.value)
        policy_joints.visible = bool(show_policy.value)
        policy_traj.visible = bool(show_policy.value)
        policy_root.visible = bool(show_policy.value)
        ref_skel.visible = bool(show_reference.value)
        ref_joints.visible = bool(show_reference.value)
        ref_traj.visible = bool(show_reference.value)
        ref_root_marker.visible = bool(show_reference.value)
        contact_points.visible = bool(show_contacts.value and contact_points.points.shape[0] > 0)
        server.flush()

    def set_play_controls() -> None:
        paused = not bool(playing.value)
        frame_slider.disabled = (not paused) or bool(state["loading"])
        prev_button.disabled = (not paused) or bool(state["loading"])
        next_button.disabled = (not paused) or bool(state["loading"])

    def set_load_controls(disabled: bool) -> None:
        motion_filter.disabled = disabled
        motion_dropdown.disabled = disabled
        search_button.disabled = disabled
        load_selected_button.disabled = disabled
        load_typed_button.disabled = disabled
        random_button.disabled = disabled
        reroll_button.disabled = disabled
        start_time_input.disabled = disabled
        max_steps_input.disabled = disabled

    def apply_bundle(bundle: RolloutBundle) -> None:
        nonlocal current_bundle, current_motion_index
        with state_lock:
            current_bundle = bundle
            current_motion_index = bundle.motion_index
            state["frame"] = 0
            state["last_tick"] = time.monotonic()
            state["loading"] = False
            ref_body_new, ref_root_new = shifted_ref(bundle, float(ref_offset.value))
            all_points_new = np.concatenate([bundle.policy_body_pos, ref_body_new], axis=1)
            grid_width_new, grid_pos_new = grid_from_points(all_points_new)
            with server.atomic():
                policy_traj.points = make_trajectory_segments(bundle.policy_root_pos)
                ref_traj.points = make_trajectory_segments(ref_root_new)
                grid_handle.width = grid_width_new
                grid_handle.height = grid_width_new
                grid_handle.position = grid_pos_new
                z_axis.points = np.asarray(
                    [[[grid_pos_new[0], grid_pos_new[1], 0.0], [grid_pos_new[0], grid_pos_new[1], 2.0]]],
                    dtype=np.float32,
                )
                z_axis_label.position = (grid_pos_new[0], grid_pos_new[1], 2.08)
                ground_label.position = (grid_pos_new[0], grid_pos_new[1], 0.02)
                frame_slider.max = bundle.num_frames - 1
                frame_slider.value = 0
                metadata.content = bundle_markdown(bundle)
                motion_filter.value = Path(bundle.motion_file).name
                load_status.content = "Loaded `{}`".format(Path(bundle.motion_file).name)
            set_frame(0)
            set_visual_toggles()
            set_play_controls()
            set_load_controls(False)
            set_client_camera(bundle)

    def load_motion_index(index: int) -> None:
        def worker() -> None:
            with state_lock:
                if bool(state["loading"]):
                    return
                state["loading"] = True
                playing.value = False
                load_status.content = "Rolling out `{}`...".format(motion_label(runner, index))
                set_play_controls()
                set_load_controls(True)
            server.flush()

            try:
                bundle = runner.generate(
                    motion_index=index,
                    start_time=float(start_time_input.value),
                    max_steps=int(max_steps_input.value),
                    stop_on_done=not bool(args.continue_after_done),
                    contact_force_threshold=float(args.contact_force_threshold),
                )
                print(
                    "rollout motion={} frames={} final_done={} mean_body_err={:.4f}".format(
                        bundle.motion_file,
                        bundle.num_frames,
                        DONE_NAMES.get(int(bundle.done_flags[-1]), str(int(bundle.done_flags[-1]))),
                        float(np.mean(bundle.body_pos_err)),
                    )
                )
            except Exception as exc:
                with state_lock:
                    state["loading"] = False
                    load_status.content = "Rollout failed: `{}`".format(exc)
                    set_play_controls()
                    set_load_controls(False)
                server.flush()
                print("rollout failed: {}: {}".format(type(exc).__name__, exc))
                return

            apply_bundle(bundle)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_motion_matches(select_first: bool = True) -> None:
        matches = motion_matches(runner, motion_filter.value, int(args.search_limit))
        if not matches:
            matches = [current_motion_index]
        options = [motion_label(runner, idx) for idx in matches]
        motion_dropdown.options = options
        if select_first or motion_dropdown.value not in options:
            motion_dropdown.value = options[0]
        load_status.content = "Showing `{}` match(es)".format(len(options))

    with server.gui.add_folder("Rollout"):
        metadata = server.gui.add_markdown(bundle_markdown(current_bundle))
        motion_filter = server.gui.add_text("Motion filter", initial_value=Path(current_bundle.motion_file).name)
        initial_matches = motion_matches(runner, Path(current_bundle.motion_file).stem, int(args.search_limit))
        if current_motion_index not in initial_matches:
            initial_matches = [current_motion_index] + initial_matches
        motion_dropdown = server.gui.add_dropdown(
            "Matches",
            options=[motion_label(runner, idx) for idx in initial_matches],
            initial_value=motion_label(runner, current_motion_index),
        )
        search_button = server.gui.add_button("Search")
        load_selected_button = server.gui.add_button("Load Selected")
        load_typed_button = server.gui.add_button("Load Typed")
        random_button = server.gui.add_button("Random")
        reroll_button = server.gui.add_button("Reroll Current")
        start_time_input = server.gui.add_number("Start time", initial_value=float(current_bundle.start_time), min=0.0, step=0.1)
        max_steps_input = server.gui.add_number("Max steps", initial_value=int(args.max_steps), min=1, step=1)
        load_status = server.gui.add_markdown("Loaded `{}`".format(Path(current_bundle.motion_file).name))

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        fps_slider = server.gui.add_slider("FPS", min=1.0, max=90.0, step=0.5, initial_value=float(args.fps))
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=current_bundle.num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        prev_button = server.gui.add_button("Prev", disabled=True)
        next_button = server.gui.add_button("Next", disabled=True)
        frame_status = server.gui.add_markdown(frame_markdown(current_bundle, 0))

    with server.gui.add_folder("Display"):
        show_policy = server.gui.add_checkbox("Policy", initial_value=True)
        show_reference = server.gui.add_checkbox("Reference", initial_value=False)
        show_contacts = server.gui.add_checkbox("Contacts", initial_value=True)
        ref_offset = server.gui.add_slider("Ref offset X", min=-2.0, max=2.0, step=0.05, initial_value=0.0)

    toggle_play = server.gui.add_command(label="Toggle Play / Pause", hotkey="space")

    @search_button.on_click
    def _(_) -> None:
        refresh_motion_matches(select_first=True)

    @load_selected_button.on_click
    def _(_) -> None:
        load_motion_index(resolve_motion_text(runner, motion_dropdown.value))

    @load_typed_button.on_click
    def _(_) -> None:
        try:
            idx = resolve_motion_text(runner, motion_filter.value)
        except Exception as exc:
            load_status.content = "Resolve failed: `{}`".format(exc)
            server.flush()
            return
        load_motion_index(idx)

    @random_button.on_click
    def _(_) -> None:
        load_motion_index(random.randrange(len(runner.motion_files)))

    @reroll_button.on_click
    def _(_) -> None:
        load_motion_index(current_motion_index)

    @playing.on_update
    def _(_) -> None:
        set_play_controls()
        state["last_tick"] = time.monotonic()

    @frame_slider.on_update
    def _(_) -> None:
        if not bool(playing.value):
            set_frame(int(frame_slider.value))

    @prev_button.on_click
    def _(_) -> None:
        set_frame((int(state["frame"]) - 1) % current_bundle.num_frames)

    @next_button.on_click
    def _(_) -> None:
        set_frame((int(state["frame"]) + 1) % current_bundle.num_frames)

    @toggle_play.on_trigger
    def _(_) -> None:
        playing.value = not bool(playing.value)

    @show_policy.on_update
    def _(_) -> None:
        set_visual_toggles()

    @show_reference.on_update
    def _(_) -> None:
        set_visual_toggles()

    @show_contacts.on_update
    def _(_) -> None:
        set_visual_toggles()

    @ref_offset.on_update
    def _(_) -> None:
        refresh_ref_offset()

    set_visual_toggles()
    set_client_camera(current_bundle)
    local_url = "http://127.0.0.1:{}".format(args.port)
    bind_url = "http://{}:{}".format(args.host, args.port)
    print("viser rollout viewer running: {} ({})".format(local_url, bind_url))

    while True:
        time.sleep(0.005)
        if bool(state["loading"]) or not bool(playing.value):
            continue
        now = time.monotonic()
        interval = 1.0 / max(float(fps_slider.value), 1e-6)
        if now - float(state["last_tick"]) < interval:
            continue
        next_frame = int(state["frame"]) + 1
        if next_frame >= current_bundle.num_frames:
            if bool(loop.value):
                next_frame = 0
            else:
                next_frame = current_bundle.num_frames - 1
                playing.value = False
        set_frame(next_frame)
        state["last_tick"] = now


def main() -> None:
    args = parse_args()
    model_file = resolve_model_file(args.model_file)
    initial_arg_file = resolve_repo_path(args.arg_file)
    print("arg_file={}".format(initial_arg_file))
    print("model_file={}".format(model_file))
    runner = RolloutRunner(
        arg_file=initial_arg_file,
        model_file=model_file,
        device=args.device,
        num_envs=args.num_envs,
        master_port=args.master_port,
    )
    motion_index = resolve_initial_motion(runner, args.motion_index, args.motion_query)
    bundle = runner.generate(
        motion_index=motion_index,
        start_time=args.start_time,
        max_steps=args.max_steps,
        stop_on_done=not bool(args.continue_after_done),
        contact_force_threshold=args.contact_force_threshold,
    )
    print(bundle_markdown(bundle).replace("`", ""))
    if args.dry_run:
        return
    serve_viewer(args, runner, bundle)


if __name__ == "__main__":
    main()
