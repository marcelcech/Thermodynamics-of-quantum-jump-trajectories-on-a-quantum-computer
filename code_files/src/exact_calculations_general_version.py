from .two_level_system_helper import *
from .exact_calculations import _calculate_single_collision
""" new general description of observable:
observable_specs = [NCOLL, depth, [O'_d, ..., O'_N]]
[O'_d, ..., O'_N] will be used periodically 
"""


def calculate_full_counting_statistics_functional(Kraus_list: list[qt.Qobj], observable_specs: list,
                                       s_val: float) -> pd.Series:
    result_dict = {'': [ket0, 1.]}
    _NCOLL = observable_specs[0]

    # calculate probabilities in a tree-like fashion
    for i in range(_NCOLL):
        new_results = {}

        for key, item in result_dict.items():
            coll_results = _calculate_single_collision(Kraus_list=Kraus_list, psi0=item[0])
            for coll_result in coll_results:
                new_results[key + str(coll_result[0])] = [coll_result[1], item[1] * coll_result[2]]

        result_dict = new_results

    # read out the probability for each traj.
    result_dict = {key: [item[1], calculate_observable_functional(key=key, observable_specs=observable_specs)] for key, item
                   in result_dict.items()}

    # bias trajectories
    df = pd.DataFrame(result_dict, index=['P(traj.)', 'O(traj.)']).T
    df['Ps(traj.)'] = df['P(traj.)'] * np.exp(-s_val * df['O(traj.)'])

    # rearrange index according to value associated to traj.
    df.sort_values('O(traj.)', inplace=True)

    # normalize probabilities
    return df['Ps(traj.)'] / df['Ps(traj.)'].sum()


def analyse_observable_histogram_functional(full_counting_statistics: pd.Series, observable_specs: list) -> pd.Series:
    # translate trajectories to ndarray
    df = pd.DataFrame(
        data=((calculate_observable_functional(key=key, observable_specs=observable_specs), item) for key, item in
              full_counting_statistics.to_dict().items()),
        columns=['O', '$P_s(O)$']
    )

    return df.groupby('O').sum()


def calculate_observable_functional(key: str, observable_specs: list) -> float:
    sum_helper = 0
    for i in range(observable_specs[1], observable_specs[0] + 1):
        sum_helper += observable_specs[2][(i - observable_specs[1]) % len(observable_specs[2])](
            [int(value) for value in key[i - observable_specs[1]:i]]
        )

    return sum_helper
