import random

def set_seed(seed: int) -> None:
    random.seed(seed)

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0