#!/usr/bin/env python3

import sys

from traj_opt.optimizer.optimizer import Optimizer
from configs import load_config


def main():
    # Check for correct number of command line args
    if len(sys.argv) < 2:
        print(
            "Error: Please provide the name of a configuration file."
        )
        return
    if len(sys.argv) > 2:
        print(
            "Error: Too many arguments. Only the name of a configuration"
            "file should be provided."
        )

    # Load the config from the given configuration file
    config = load_config(sys.argv[1])

    # Construct the optimizer using given config
    optimizer = Optimizer(config)

    # Solve the OCP and display the result
    optimizer.solve()


if __name__ == "__main__":
    main()