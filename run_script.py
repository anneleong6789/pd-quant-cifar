import os
import argparse
import time
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_name",
        type=str,
        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'],
        help="Experiment/model name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/data",
        help="Path to ImageNet dataset root (should contain 'train' and 'val')"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save experiment results"
    )
    args = parser.parse_args()

    # Validate dataset path
    if not os.path.exists(os.path.join(args.data_path, "train")):
        sys.exit(f"❌ Dataset not found at {args.data_path}/train. Please set --data_path correctly.")

    # Create results folder
    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]

    arch = args.exp_name  # same naming

    # Hyperparams depending on model
    if arch in ["resnet18", "resnet50"]:
        weight, T, lamb_c = 0.01, 4.0, 0.02
    elif arch in ["regnetx_600m", "regnetx_3200m"]:
        weight, T, lamb_c = 0.01, 4.0, 0.01
    elif arch == "mobilenetv2":
        weight, T, lamb_c = 0.1, 1.0, 0.005
    elif arch == "mnasnet":
        weight, T, lamb_c = 0.2, 1.0, 0.001
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # Loop through quantization configs
    for i in range(4):
        cmd = (
            f"python main_imagenet.py "
            f"--data_path \"{args.data_path}\" "
            f"--arch {arch} "
            f"--n_bits_w {w_bits[i]} "
            f"--n_bits_a {a_bits[i]} "
            f"--weight {weight} "
            f"--T {T} "
            f"--lamb_c {lamb_c} "
            f"--exp_name {args.exp_name} "
            f"--results_dir {exp_dir}"
        )
        print(f"\n🚀 Running: {cmd}\n")
        os.system(cmd)
        time.sleep(0.5)

