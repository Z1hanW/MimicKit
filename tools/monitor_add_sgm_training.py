#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MONITOR_DIR = ROOT_DIR / "output" / "add_sgm_monitor"
STATE_FILE = MONITOR_DIR / "state.json"
EVENT_LOG = MONITOR_DIR / "monitor_events.log"

TRAIN_SESSION = "sgm_add_8gpu_5day"
DEFAULT_TARGET_SAMPLES = 14_000_000_000
DEFAULT_MAX_INTERVENTIONS = 2
OUTPUT_INTERVAL_SAMPLES = 4096 * 32 * 200
MIN_STAGNANT_CHECK_SECONDS = 5400
STARTUP_GRACE_SECONDS = 600
STRICT_READY_EPISODE_STEPS = 180.0
ROOT_TIGHT_MIN_EPISODE_STEPS = 90.0

PROFILES = {
    "track": {
        "arg_file": "args/add_smpl_generalsit_heightmap_track_args.txt",
        "wandb_prefix": "add-sgm-8gpu-track",
    },
    "track_curriculum": {
        "arg_file": "args/add_smpl_generalsit_heightmap_track_curriculum_args.txt",
        "wandb_prefix": "add-sgm-8gpu-track-curriculum",
    },
    "track_root_tight": {
        "arg_file": "args/add_smpl_generalsit_heightmap_track_root_tight_args.txt",
        "wandb_prefix": "add-sgm-8gpu-track-root-tight",
    },
    "track_strict": {
        "arg_file": "args/add_smpl_generalsit_heightmap_track_strict_args.txt",
        "wandb_prefix": "add-sgm-8gpu-track-strict",
    },
}

INIT_STATE = {
    "active_out_dir": "output/add_sgm_8gpu_track_5day_from_1e8",
    "profile": "track",
    "base_completed_samples": 0,
    "active_max_samples": DEFAULT_TARGET_SAMPLES,
    "target_total_samples": DEFAULT_TARGET_SAMPLES,
    "last_active_samples": 0,
    "last_check_ts": 0.0,
    "stagnant_checks": 0,
    "interventions": 0,
    "restarts": 0,
}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)


def load_state():
    ensure_dirs()
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = dict(INIT_STATE)
        save_state(state)
    for key, value in INIT_STATE.items():
        state.setdefault(key, value)
    return state


def save_state(state):
    ensure_dirs()
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(STATE_FILE)


def log_event(event):
    ensure_dirs()
    event = dict(event)
    event.setdefault("time", utc_now())
    line = json.dumps(event, sort_keys=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_cmd(args, check=False):
    return subprocess.run(
        args,
        cwd=str(ROOT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def run_shell(cmd, check=False):
    return subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        executable="/bin/bash",
    )


def parse_log(out_dir):
    log_file = ROOT_DIR / out_dir / "log.txt"
    if not log_file.exists():
        return []

    lines = [line.strip() for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        vals = line.split()
        if len(vals) != len(header):
            continue
        row = {}
        ok = True
        for key, val in zip(header, vals):
            try:
                if key in ("Iteration", "Samples", "Test_Episodes", "Train_Episodes"):
                    row[key] = int(float(val))
                else:
                    row[key] = float(val)
            except ValueError:
                ok = False
                break
        if ok:
            rows.append(row)
    return rows


def find_run_url(out_dir):
    rank0 = ROOT_DIR / out_dir / "rank_0.log"
    if not rank0.exists():
        return None
    text = rank0.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+", text)
    return matches[-1] if matches else None


def process_alive(out_dir):
    proc = run_shell("pgrep -af 'mimickit/run.py' || true")
    needle = "--out_dir " + out_dir
    return needle in proc.stdout, proc.stdout


def recent_errors(out_dir):
    error_patterns = [
        "Traceback",
        "RuntimeError",
        "OutOfMemoryError",
        "out of memory",
        "illegal memory access",
        "CUDA error",
        "NCCL error",
        "Exception",
    ]
    messages = []
    for log_file in sorted((ROOT_DIR / out_dir).glob("rank_*.log")):
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tail = text[-20000:]
        for pattern in error_patterns:
            if pattern in tail:
                messages.append(f"{log_file.name}: {pattern}")
                break
    return messages


def has_bad_numbers(rows):
    for row in rows[-5:]:
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return True, key
    return False, None


def latest_checkpoint(out_dir):
    candidates = []
    model = ROOT_DIR / out_dir / "model.pt"
    if model.exists():
        candidates.append(model)
    int_dir = ROOT_DIR / out_dir / "int_models"
    if int_dir.exists():
        candidates.extend(int_dir.glob("model_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def disk_free_gb():
    usage = shutil.disk_usage(ROOT_DIR)
    return usage.free / (1024 ** 3)


def out_dir_age_seconds(out_dir):
    path = ROOT_DIR / out_dir
    if not path.exists():
        return 0.0
    return max(0.0, time.time() - path.stat().st_mtime)


def current_master_port():
    return str(6200 + int(time.time()) % 700)


def tmux_has_session(name):
    return run_cmd(["tmux", "has-session", "-t", name]).returncode == 0


def stop_training_session(out_dir):
    if tmux_has_session(TRAIN_SESSION):
        run_cmd(["tmux", "kill-session", "-t", TRAIN_SESSION])
    # Give the launcher a moment to propagate SIGHUP, then clean up any stragglers.
    time.sleep(3)
    pattern = f"mimickit/run.py .*--out_dir {re.escape(out_dir)}"
    run_shell(f"pkill -f {shlex.quote(pattern)} || true")
    run_shell("pkill -f 'tools/launch_add_sgm_8gpu_external.sh' || true")


def start_training(profile, out_dir, model_file, max_samples):
    profile_cfg = PROFILES[profile]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wandb_name = f"{profile_cfg['wandb_prefix']}-auto-{timestamp}"
    master_port = current_master_port()
    quoted = {
        "root": shlex.quote(str(ROOT_DIR)),
        "arg_file": shlex.quote(profile_cfg["arg_file"]),
        "out_dir": shlex.quote(out_dir),
        "model_file": shlex.quote(str(model_file)),
        "wandb_name": shlex.quote(wandb_name),
        "max_samples": shlex.quote(str(max_samples)),
        "master_port": shlex.quote(master_port),
    }
    cmd = (
        "cd {root} && mkdir -p {out_dir} && "
        "ARG_FILE={arg_file} OUT_DIR={out_dir} MODEL_FILE={model_file} "
        "WANDB_NAME={wandb_name} NUM_ENVS=512 MAX_SAMPLES={max_samples} "
        "MASTER_PORT={master_port} SAVE_INT_MODELS=true "
        "tools/launch_add_sgm_8gpu_external.sh > {out_dir}/launcher.log 2>&1"
    ).format(**quoted)
    run_cmd(["tmux", "new-session", "-d", "-s", TRAIN_SESSION, cmd], check=True)
    return {
        "wandb_name": wandb_name,
        "master_port": master_port,
        "command": cmd,
    }


def summarize(rows, state, out_dir):
    last = rows[-1] if rows else {}
    active_samples = int(last.get("Samples", 0)) if last else 0
    total_samples = int(state.get("base_completed_samples", 0)) + active_samples
    summary = {
        "out_dir": out_dir,
        "profile": state.get("profile"),
        "rows": len(rows),
        "active_samples": active_samples,
        "total_samples": total_samples,
        "target_total_samples": int(state.get("target_total_samples", DEFAULT_TARGET_SAMPLES)),
        "iteration": last.get("Iteration"),
        "samples_per_second": last.get("Samples_Per_Second"),
        "test_episode_length": last.get("Test_Episode_Length"),
        "train_episode_length": last.get("Train_Episode_Length"),
        "root_pos_err": last.get("Root_Pos_Err"),
        "body_pos_err": last.get("Body_Pos_Err"),
        "dof_vel_err": last.get("Dof_Vel_Err"),
        "root_vel_err": last.get("Root_Vel_Err"),
        "disc_reward_mean": last.get("Disc_Reward_Mean"),
        "critic_loss": last.get("Critic_Loss"),
    }
    return summary


def metric_delta(rows, key, window=5):
    if len(rows) < 2:
        return 0.0
    recent = rows[-window:]
    if len(recent) < 2 or key not in recent[0] or key not in recent[-1]:
        return 0.0
    return recent[-1][key] - recent[0][key]


def decide(rows, state, alive, errors):
    out_dir = state["active_out_dir"]
    summary = summarize(rows, state, out_dir)
    active_samples = summary["active_samples"]
    total_samples = summary["total_samples"]
    target_total = summary["target_total_samples"]

    if total_samples >= target_total:
        return "complete", "target_total_samples_reached", summary

    if errors:
        return "restart_same", "rank_log_error: " + "; ".join(errors[:3]), summary

    bad_numbers, bad_key = has_bad_numbers(rows)
    if bad_numbers:
        return "restart_same", f"bad_number_in_{bad_key}", summary

    if not alive and active_samples == 0 and len(rows) == 0 and out_dir_age_seconds(out_dir) < STARTUP_GRACE_SECONDS:
        return "healthy", "waiting_for_training_process_startup_or_first_log", summary

    if not alive and active_samples < state.get("active_max_samples", DEFAULT_TARGET_SAMPLES):
        return "restart_same", "training_process_not_alive", summary

    last_active_samples = int(state.get("last_active_samples", 0))
    now_ts = time.time()
    last_check_ts = float(state.get("last_check_ts", 0.0) or 0.0)
    elapsed_since_last_check = now_ts - last_check_ts if last_check_ts > 0 else None
    stagnant_check_eligible = (
        elapsed_since_last_check is not None
        and elapsed_since_last_check >= MIN_STAGNANT_CHECK_SECONDS
    )

    if active_samples <= last_active_samples and alive and active_samples > 0:
        if stagnant_check_eligible:
            state["stagnant_checks"] = int(state.get("stagnant_checks", 0)) + 1
    else:
        state["stagnant_checks"] = 0
    state["last_active_samples"] = active_samples
    state["last_check_ts"] = now_ts

    if state["stagnant_checks"] >= 2:
        return "restart_same", "samples_stagnant_for_two_checks", summary

    if disk_free_gb() < 10:
        return "observe", "low_disk_space_under_10gb_manual_cleanup_needed", summary

    enough_quality_data = active_samples >= 250_000_000 and len(rows) >= 4
    if enough_quality_data:
        profile = state.get("profile", "track")
        root = summary.get("root_pos_err")
        body = summary.get("body_pos_err")
        ep_len = summary.get("test_episode_length")
        root_delta = metric_delta(rows, "Root_Pos_Err")
        body_delta = metric_delta(rows, "Body_Pos_Err")
        ep_delta = metric_delta(rows, "Test_Episode_Length")

        if profile == "track_curriculum":
            if root is not None and ep_len is not None and root < 0.30 and ep_len > STRICT_READY_EPISODE_STEPS:
                return "adjust_strict", f"curriculum_ready root={root:.4f} ep={ep_len:.4f}", summary
            if (
                active_samples >= 750_000_000
                and root is not None
                and ep_len is not None
                and root > 0.55
                and ep_len > ROOT_TIGHT_MIN_EPISODE_STEPS
                and root_delta > -0.015
            ):
                return "adjust_root_tight", f"curriculum_root_drift root={root:.4f} delta={root_delta:.4f} ep={ep_len:.4f}", summary
            return "healthy", "curriculum_progressing_or_collecting_signal", summary

        if profile == "track_root_tight":
            if root is not None and ep_len is not None and root < 0.30 and ep_len > STRICT_READY_EPISODE_STEPS:
                return "adjust_strict", f"root_tight_ready root={root:.4f} ep={ep_len:.4f}", summary
            return "healthy", "root_tight_progressing_or_collecting_signal", summary

        if root is not None and root > 0.45 and profile != "track_strict":
            return "adjust_strict", f"root_pos_err_too_high root={root:.4f} delta={root_delta:.4f}", summary
        if root is not None and root > 0.75 and root_delta > 0.07:
            return "adjust_strict", f"root_pos_err_drifting root={root:.4f} delta={root_delta:.4f}", summary
        if body is not None and body > 0.18 and body_delta > 0.035:
            return "adjust_strict", f"body_pos_err_drifting body={body:.4f} delta={body_delta:.4f}", summary
        if ep_len is not None and ep_len < 18.0 and ep_delta < -8.0:
            return "adjust_strict", f"episode_length_collapsing ep={ep_len:.4f} delta={ep_delta:.4f}", summary

    return "healthy", "progressing_within_expected_bounds", summary


def remaining_samples(state, active_samples):
    target = int(state.get("target_total_samples", DEFAULT_TARGET_SAMPLES))
    base = int(state.get("base_completed_samples", 0))
    remaining = target - base - int(active_samples)
    return max(OUTPUT_INTERVAL_SAMPLES, remaining)


def make_new_out_dir(profile, reason):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_]+", "_", reason)[:48].strip("_")
    return f"output/add_sgm_8gpu_{profile}_auto_{safe_reason}_{stamp}"


def maybe_intervene(action, reason, summary, state):
    if action not in ("restart_same", "adjust_strict", "adjust_root_tight"):
        return None

    active_out_dir = state["active_out_dir"]
    checkpoint = latest_checkpoint(active_out_dir)
    if checkpoint is None:
        log_event({
            "event": "intervention_blocked",
            "action": action,
            "reason": reason,
            "detail": "no_checkpoint_found",
            "summary": summary,
        })
        return None

    active_samples = int(summary.get("active_samples") or 0)
    state["base_completed_samples"] = int(state.get("base_completed_samples", 0)) + active_samples
    state["last_active_samples"] = 0
    state["stagnant_checks"] = 0

    if action == "adjust_strict":
        if state.get("profile") == "track_strict" or int(state.get("interventions", 0)) >= DEFAULT_MAX_INTERVENTIONS:
            log_event({
                "event": "adjustment_deferred",
                "action": action,
                "reason": reason,
                "detail": "already_strict_or_intervention_limit_reached",
                "summary": summary,
            })
            save_state(state)
            return None
        new_profile = "track_strict"
        state["interventions"] = int(state.get("interventions", 0)) + 1
    elif action == "adjust_root_tight":
        if state.get("profile") == "track_root_tight" or int(state.get("interventions", 0)) >= DEFAULT_MAX_INTERVENTIONS:
            log_event({
                "event": "adjustment_deferred",
                "action": action,
                "reason": reason,
                "detail": "already_root_tight_or_intervention_limit_reached",
                "summary": summary,
            })
            save_state(state)
            return None
        new_profile = "track_root_tight"
        state["interventions"] = int(state.get("interventions", 0)) + 1
    else:
        new_profile = state.get("profile", "track")
        state["restarts"] = int(state.get("restarts", 0)) + 1

    new_out_dir = make_new_out_dir(new_profile, action)
    max_samples = remaining_samples(state, 0)

    log_event({
        "event": "intervention_start",
        "action": action,
        "reason": reason,
        "old_out_dir": active_out_dir,
        "new_out_dir": new_out_dir,
        "checkpoint": str(checkpoint.relative_to(ROOT_DIR)),
        "new_profile": new_profile,
        "remaining_samples": max_samples,
    })

    stop_training_session(active_out_dir)
    launch_info = start_training(new_profile, new_out_dir, checkpoint.relative_to(ROOT_DIR), max_samples)

    state["active_out_dir"] = new_out_dir
    state["profile"] = new_profile
    state["active_max_samples"] = max_samples
    state["last_launch"] = launch_info
    state["last_intervention_reason"] = reason
    save_state(state)

    log_event({
        "event": "intervention_done",
        "action": action,
        "new_out_dir": new_out_dir,
        "new_profile": new_profile,
        "wandb_name": launch_info["wandb_name"],
        "master_port": launch_info["master_port"],
    })
    return launch_info


def check_once():
    state = load_state()
    out_dir = state["active_out_dir"]
    rows = parse_log(out_dir)
    alive, process_text = process_alive(out_dir)
    errors = recent_errors(out_dir)
    action, reason, summary = decide(rows, state, alive, errors)
    summary["alive"] = alive
    summary["disk_free_gb"] = round(disk_free_gb(), 2)
    summary["wandb_url"] = find_run_url(out_dir)

    log_event({
        "event": "check",
        "action": action,
        "reason": reason,
        "summary": summary,
    })

    save_state(state)
    launch_info = maybe_intervene(action, reason, summary, state)
    if launch_info is None:
        save_state(state)
    return action


def main():
    parser = argparse.ArgumentParser(description="Monitor and conservatively recover the SGM 8-GPU ADD tracking run.")
    parser.add_argument("--interval-sec", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        check_once()
        return

    log_event({"event": "monitor_started", "interval_sec": args.interval_sec})
    while True:
        try:
            check_once()
        except Exception as exc:
            log_event({"event": "monitor_exception", "error": repr(exc)})
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
