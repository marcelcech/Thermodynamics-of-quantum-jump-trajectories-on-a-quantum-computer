from .two_level_system_helper import *


def calculate_full_counting_statistics(Kraus_list: list[qt.Qobj], biasing_pattern: np.ndarray,
                                       s_val: float) -> pd.Series:
    result_dict = {'': [ket0, 1.]}
    _NCOLL = biasing_pattern.size

    # calculate probabilities in a tree-like fashion
    for i in range(_NCOLL):
        new_results = {}

        for key, item in result_dict.items():
            coll_results = _calculate_single_collision(Kraus_list=Kraus_list, psi0=item[0])
            for coll_result in coll_results:
                new_results[key + str(coll_result[0])] = [coll_result[1], item[1] * coll_result[2]]

        result_dict = new_results

    # read out the probability for each traj.
    result_dict = {key: [item[1], calculate_observable(key=key, biasing_pattern=biasing_pattern)] for key, item
                   in result_dict.items()}

    # bias trajectories
    df = pd.DataFrame(result_dict, index=['P(traj.)', 'O(traj.)']).T
    df['Ps(traj.)'] = df['P(traj.)'] * np.exp(-s_val * df['O(traj.)'])

    # rearrange index according to value associated to traj.
    df.sort_values('O(traj.)', inplace=True)

    # normalize probabilities
    return df['Ps(traj.)'] / df['Ps(traj.)'].sum()


def analyse_observable_histogram(full_counting_statistics: pd.Series, biasing_pattern: np.ndarray) -> pd.Series:
    # translate trajectories to ndarray
    df = pd.DataFrame(
        data=((calculate_observable(key=key, biasing_pattern=biasing_pattern), item) for key, item in
              full_counting_statistics.to_dict().items()),
        columns=['O', '$P_s(O)$']
    )

    return df.groupby('O').sum()


def analyse_temporal_probability(full_counting_statistics: pd.Series) -> np.ndarray:
    # calculate probabilities for '1' outcome at every collision
    helper_list = []
    for index in full_counting_statistics.index:
        helper_list.append(list(index))
    data = np.array(helper_list, dtype=float)

    weight = full_counting_statistics.copy().values
    weight = weight.reshape((weight.size, 1))
    data = weight * data

    return data.sum(axis=0)


def _calculate_single_collision(Kraus_list: list[qt.Qobj], psi0: qt.Qobj) -> list[list[int, qt.Qobj, float]]:
    result_list = []
    i_jump = 0

    for op in Kraus_list:
        psi = op * psi0
        result_list.append([i_jump, psi / psi.norm(), psi.norm() ** 2])
        i_jump += 1

    return result_list


def calculate_observable(key: str, biasing_pattern: np.ndarray) -> float:
    sum_helper = 0
    for i, value in enumerate(key):
        sum_helper += biasing_pattern[i] * int(value)

    return sum_helper
