from pathlib import Path
import argparse
import numpy as np


def format_array(name, value, max_rows=5, max_cols=8):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{name}: {array.item()}"
    if array.ndim == 1:
        preview = array[:max_cols]
        suffix = " ..." if array.shape[0] > max_cols else ""
        return f"{name}: shape={array.shape}, first={preview.tolist()}{suffix}"
    rows = min(array.shape[0], max_rows)
    cols = min(array.shape[1], max_cols)
    preview = array[:rows, :cols]
    suffix = " ..." if array.shape[1] > max_cols or array.shape[0] > max_rows else ""
    return f"{name}: shape={array.shape}, preview={preview.tolist()}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Read and print an STS npz log.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "logs" / "sts_muscle_log.npz"),
        help="Path to the npz log file.",
    )
    args = parser.parse_args()

    log_path = Path(args.path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    data = np.load(log_path, allow_pickle=True)
    print(f"Loaded: {log_path}")
    print("Keys:", ", ".join(data.files))

    for key in data.files:
        print(format_array(key, data[key]))

    if "actuator_names" in data.files:
        names = data["actuator_names"]
        print("\nActuator names:")
        for index, name in enumerate(names):
            print(f"  {index}: {name}")

    if "act" in data.files and data["act"].ndim == 2:
        act = data["act"]
        print("\nActivation summary:")
        print(f"  frames: {act.shape[0]}")
        print(f"  muscles/actuators: {act.shape[1]}")
        print(f"  max activation per column: {np.max(act, axis=0).tolist()}")

    if "actuator_force" in data.files and data["actuator_force"].ndim == 2:
        force = data["actuator_force"]
        print("\nForce summary:")
        print(f"  frames: {force.shape[0]}")
        print(f"  actuators: {force.shape[1]}")
        print(f"  max abs force per column: {np.max(np.abs(force), axis=0).tolist()}")


if __name__ == "__main__":
    main()
