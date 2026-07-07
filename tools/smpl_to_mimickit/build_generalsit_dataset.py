import argparse
import csv
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.smpl_to_mimickit.smpl_to_mimickit import convert_smpl_to_mimickit


def read_motion_list(motion_list_file):
    motion_entries = []
    with open(motion_list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            motion_entries.append(line)
    return motion_entries


def read_motion_manifest(manifest_file, input_column):
    motion_entries = []
    with open(manifest_file, "r") as f:
        reader = csv.DictReader(f)
        if input_column not in reader.fieldnames:
            raise ValueError("Manifest is missing column: {}".format(input_column))

        for row in reader:
            motion_entries.append(row[input_column])

    return motion_entries


def path_for_yaml(path):
    return path.as_posix()


def main():
    parser = argparse.ArgumentParser(description="Build a MimicKit SMPL dataset from BEDLAM z-up npz files.")
    parser.add_argument("--input_dir", default="", help="Directory containing z-up SMPL npz files.")
    parser.add_argument("--motion_list", default="", help="Text file with one npz filename per line.")
    parser.add_argument("--manifest_csv", default="", help="CSV manifest with one motion per row.")
    parser.add_argument("--manifest_input_column", default="zup_source", help="Manifest column containing input npz paths or names.")
    parser.add_argument("--output_dir", default="data/motions/smpl_generalsit", help="Output directory for MimicKit pkl motions.")
    parser.add_argument("--dataset_file", default="data/datasets/dataset_smpl_generalsit.yaml", help="Output dataset yaml file.")
    parser.add_argument("--loop", default="clamp", choices=["wrap", "clamp"], help="Loop mode for converted motions.")
    parser.add_argument("--output_fps", type=int, default=-1, help="Output frame rate. -1 keeps source fps.")
    parser.add_argument("--z_correction", default="none", choices=["none", "calibrate", "full"], help="Z correction passed to converter.")
    parser.add_argument("--input_coordinate_system", default="auto", choices=["auto", "y-up", "z-up"], help="Input coordinate system.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted pkl files.")
    parser.add_argument("--dry_run", action="store_true", help="Only write the dataset yaml, without converting pkl files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir != "" else None
    output_dir = Path(args.output_dir)
    dataset_file = Path(args.dataset_file)
    if args.manifest_csv != "":
        motion_entries = read_motion_manifest(args.manifest_csv, args.manifest_input_column)
    elif args.motion_list != "":
        motion_entries = read_motion_list(args.motion_list)
    else:
        raise ValueError("Either --motion_list or --manifest_csv must be provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    motions = []
    for motion_entry in motion_entries:
        input_file = Path(motion_entry)
        if not input_file.is_absolute():
            if input_dir is None:
                raise ValueError("Relative motion entry requires --input_dir: {}".format(motion_entry))
            input_file = input_dir / motion_entry

        output_file = output_dir / (input_file.stem + ".pkl")

        if not input_file.exists():
            raise FileNotFoundError(input_file)

        if args.overwrite or not output_file.exists():
            if not args.dry_run:
                convert_smpl_to_mimickit(
                    str(input_file),
                    str(output_file),
                    loop_mode=args.loop,
                    output_fps=args.output_fps,
                    z_correction=args.z_correction,
                    input_coordinate_system=args.input_coordinate_system,
                )

        motions.append({
            "file": path_for_yaml(output_file),
            "weight": 1.0,
        })

    with open(dataset_file, "w") as f:
        yaml.safe_dump({"motions": motions}, f, sort_keys=False)

    print("Wrote {} motions to {}".format(len(motions), dataset_file))
    if args.dry_run:
        print("Dry run: pkl conversion was skipped.")


if __name__ == "__main__":
    main()
