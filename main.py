#!/usr/bin/env python3

from traj_opt.optimizer.optimizer import Optimizer
from traj_opt.config import TrajOptConfig

def main():
    # Construct the config
    config = TrajOptConfig()

    # Construct the optimizer using given config
    optimizer = Optimizer(config)

    # Solve the OCP and display the result
    optimizer.solve()


if __name__ == "__main__":
    main()