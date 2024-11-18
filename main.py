#!/usr/bin/env python3

from traj_opt.optimizer.optimizer import Optimizer
from traj_opt.config import TrajOptConfig

def main():
    config = TrajOptConfig()
    optimizer = Optimizer(config)

    optimizer.solve()


if __name__ == "__main__":
    main()