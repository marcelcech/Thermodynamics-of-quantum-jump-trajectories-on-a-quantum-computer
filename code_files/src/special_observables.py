from .package_requirements import *


def magnetization_pattern(NCOLL: int) -> np.ndarray:
    return np.ones(NCOLL)


def imbalance_pattern(NCOLL: int) -> np.ndarray:
    helper = np.ones(NCOLL)
    helper[::2] = -1

    return helper


def magnetization_specs(NCOLL: int) -> list:
    return [NCOLL, 1, [lambda x: x[0]]]


def correlation_specs(NCOLL: int) -> list:
    return [NCOLL, 2, [lambda x: 4 * x[0] * x[1] - 2 * x[0] - 2 * x[1] + 1]]


def dummy_test_specs(NCOLL: int) -> list:
    return [NCOLL, 2, [lambda x: -2 * x[0] + x[1] + 1, lambda x: x[0]]]


def three_body_correlation_specs(NCOLL: int) -> list:
    return [NCOLL, 3, [lambda x: -8 * x[0] * x[1] * x[2] + 4 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2]) - 2 * (
            x[0] + x[1] + x[2]) + 1]]
