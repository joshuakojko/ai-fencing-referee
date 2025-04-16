import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    array = np.load(f"fencing_dataset/pose_data/{args.path}.npy")
    print(array)


if __name__ == "__main__":
    main()
