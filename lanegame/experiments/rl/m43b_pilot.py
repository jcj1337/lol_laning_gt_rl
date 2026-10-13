"""`python -m lanegame experiments-rl m43b_pilot`: every stage of the M4.3b pilot (lanegame.experiments.rl.m43b)."""

from __future__ import annotations

from .m43b.__main__ import main


def run() -> None:
    main(["all"])


if __name__ == "__main__":
    main()
