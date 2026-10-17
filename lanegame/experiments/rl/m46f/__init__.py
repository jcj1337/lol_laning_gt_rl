"""M4.6f: separate-critic continuation pilot (docs/m4/m4_6/m4_6f_preregistration.md).

Matched continuations of the six M4.6c 8.192M C1 policies: the unchanged shared actor-critic network (control) against
the same actor with a separately trainable critic trunk copied from the shared trunk (intervention). Run as
`python -m lanegame.experiments.rl.m46f <stage> [...]`; see `__main__`.
"""
