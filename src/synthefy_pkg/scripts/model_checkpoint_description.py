import sys

import torch

COMPILE = False


def main(model_path):
    model = torch.load(model_path)
    print(f"epoch: {model['epoch']}, global_step: {model['global_step']}")


if __name__ == "__main__":
    model_path = sys.argv[1]
    main(model_path)
