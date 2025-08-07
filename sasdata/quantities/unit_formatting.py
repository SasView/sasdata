
import numpy as np


def solve_contributions(target: float, scales: list[float], max_power: int=4, tol=1e-5):
    log_target = np.log10(target)
    log_scale_pairs = sorted([(i, np.log10(scale)) for i, scale in enumerate(scales)], key=lambda x: x[1])

    ordering = [i for i, _ in log_scale_pairs]
    log_scale = [l for _, l in log_scale_pairs]

    powers = [0 for _ in scales]

