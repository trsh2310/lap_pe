import argparse
import subprocess
import sys
from pathlib import Path


def read_max_time_interval_choices(params_file: Path) -> list[int]:
    lines = params_file.read_text().splitlines()
    in_param = False
    in_choices = False
    param_type = None
    choices = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- name:"):
            if in_param:
                break
            in_param = stripped.split(":", 1)[1].strip() == "max_time_interval"
            continue
        if not in_param:
            continue
        if stripped.startswith("type:"):
            param_type = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("choices:"):
            in_choices = True
            continue
        if in_choices and stripped.startswith("- "):
            choices.append(int(stripped[2:].strip()))
            continue
        if in_choices and stripped and not stripped.startswith("- "):
            break

    if not choices:
        raise ValueError(f"max_time_interval choices are not defined in {params_file}.")
    if param_type != "categorical":
        raise ValueError("max_time_interval must be categorical to run this grid script.")
    return choices


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run TiSASRec for every max_time_interval listed in "
            "configs/optuna_params/tisasrec.yaml, while keeping other model "
            "parameters fixed from the Hydra config."
        )
    )
    parser.add_argument("--config_name", "-cn", default="tisasrec")
    parser.add_argument("--dataset", "-ds", default=None)
    parser.add_argument(
        "--params_file",
        "-pf",
        default="configs/optuna_params/tisasrec.yaml",
        help=(
            "YAML file containing categorical max_time_interval choices. "
            "Ignored when --values is provided."
        ),
    )
    parser.add_argument(
        "--values",
        "-v",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit max_time_interval values. This avoids the params file, "
            "for example: --values 16 64 256 512 1024."
        ),
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to run run_model.py.",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.values is not None:
        choices = args.values
    else:
        params_file = repo_root / args.params_file
        choices = read_max_time_interval_choices(params_file)

    if not choices:
        parser.error("At least one max_time_interval value is required.")
    if any(value < 0 for value in choices):
        parser.error("max_time_interval values must be non-negative.")

    print(f"max_time_interval choices: {choices}")
    for value in choices:
        overrides = [
            f"model.max_time_interval={value}",
            f"model.name=tisasrec_mti_{value}",
        ]
        if args.dataset:
            overrides.append(f"dataset={args.dataset}")
        overrides.extend(args.override)

        cmd = [
            args.python_bin,
            "run_model.py",
            "-cn",
            args.config_name,
            *overrides,
        ]
        print("\nRunning:", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
