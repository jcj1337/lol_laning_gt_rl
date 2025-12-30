import random

def set_seed(seed: int) -> None:
    random.seed(seed)


def clip(x, low, high):
    return low if x < low else high if x > high else x


def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0