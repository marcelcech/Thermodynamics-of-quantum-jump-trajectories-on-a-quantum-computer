import numpy as np
import pandas as pd

from .package_requirements import *
from .two_level_system_helper import ket1, ket0
from .simulation_handler import read_data_from_dict, simulate_quantum_circuits
from .exact_calculations import calculate_observable


class TimeDependentDoob:
    def __init__(self, collision_unitary: qt.Qobj, biasing_pattern: np.ndarray, s_val: float):
        self._backend = None
        self._method = None
        self._NTRAJ = None
        self._U = collision_unitary
        self._biasing_pattern = biasing_pattern
        self._NCOLL = biasing_pattern.size
        self._s_val = s_val

        # calculate the Kraus operators for the unbiased dynamics
        self._undoobed_Kraus_ops = [qt.tensor(bra, qt.qeye(2)) * self._U * qt.tensor(ket0, qt.qeye(2)) for bra in
                                    [ket0.dag(), ket1.dag()]]

        # calculate the Doob-transform
        _biased_maps = [sum([np.exp(-i * self._s_val * p_n) * qt.sprepost(K, K.dag()) for i, K in
                             enumerate(self._undoobed_Kraus_ops)]) for p_n
                        in self._biasing_pattern]

        _G_n_list = [qt.qeye(2)]
        for n in np.arange(self._NCOLL - 1, -1, -1):
            _G_n_list.append((_biased_maps[n].dag()(_G_n_list[-1] ** 2)).sqrtm())

        _G_n_list = _G_n_list[::-1]
        _inv_G_n_list = [G.inv() for G in _G_n_list]

        self._Utilde = [None]
        for n in range(1, self._NCOLL + 1):
            _tilted_Kraus_operators = [
                np.exp(-i * self._s_val * self._biasing_pattern[n - 1] / 2) * _G_n_list[n] *
                self._undoobed_Kraus_ops[i] * _inv_G_n_list[n - 1] for i in range(2)]

            # test for trace-preserving property
            assert (sum([K.dag() * K for K in _tilted_Kraus_operators]) - qt.qeye(2)).full().max() < 1e-10

            # calculate new collision unitary
            kets = [ket0, ket1]
            U_helper = np.zeros((4, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U_helper[2 * i + j, k] = (kets[j].dag() * _tilted_Kraus_operators[i] * kets[k]).full()[0][0]

            for i in range(2):
                # # general but unpredicted filling
                # U_helper = np.append(U_helper, find_orthogonal_vec(U_helper).reshape((4, 1)), axis=1)
                # U_helper[:, 2 + i] = U_helper[:, 2 + i] / np.linalg.full().max(U_helper[:, 2 + i])

                # exploit particle-hole symmetry
                U_helper = np.append(U_helper, U_helper[::-1, -1 - 2 * i].reshape((4, 1)), axis=1)

            # check unitary
            assert (qt.Qobj(U_helper) * qt.Qobj(U_helper).dag() - qt.qeye(4)).full().max() < 1e-10

            self._Utilde.append(U_helper)

        # calculate new initial state
        _psi_tilde_0 = (_G_n_list[0] * ket0).unit()
        self._initial_state = [x[0] for x in _psi_tilde_0.full().tolist()]

    def get_undoobed_Kraus_ops(self) -> list[qt.Qobj]:
        return copy(self._undoobed_Kraus_ops)

    def get_possible_observable_values(self) -> np.ndarray:
        _possible_vals = [0]
        for p_n in self._biasing_pattern:
            _copy_possible_vals = copy(_possible_vals)
            _possible_vals = []
            for current_val in _copy_possible_vals:
                for i in range(2):
                    _possible_vals.append(current_val + i * p_n)

            _possible_vals = np.unique(_possible_vals)

        return np.array(sorted(_possible_vals))

    def _get_bins_for_obs_hist(self) -> np.ndarray:
        return np.append(self.get_possible_observable_values(), np.max(self.get_possible_observable_values()) + 1)

    def get_undoobed_quantum_circuit(self, name: str = 'None') -> qiskit.QuantumCircuit:
        # create quantum circuit
        _qr = qiskit.QuantumRegister(self._NCOLL + 1)
        _cr = qiskit.ClassicalRegister(self._NCOLL, 'trajs')
        quantum_circuit = qiskit.QuantumCircuit(_qr, _cr)
        quantum_circuit.name = name

        # initialize the system at the first place
        quantum_circuit.initialize(self._initial_state, 0)

        # add collisions and measurements
        for n in range(1, self._NCOLL + 1):
            quantum_circuit.append(
                qiskit.extensions.UnitaryGate(self._U.full(), label=r'$U$'),
                qargs=[0, self._NCOLL + 1 - n]
            )
            quantum_circuit.measure(qubit=self._NCOLL + 1 - n, cbit=n - 1)

        return quantum_circuit

    def get_doobed_quantum_circuit(self, name: str = 'None') -> qiskit.QuantumCircuit:
        #  return undoobed circuit, if s=0
        if np.abs(self._s_val) < 1e-9:
            return self.get_undoobed_quantum_circuit(name=name)

        # otherwise create doobed quantum circuit
        _qr = qiskit.QuantumRegister(self._NCOLL + 1)
        _cr = qiskit.ClassicalRegister(self._NCOLL, 'trajs')
        quantum_circuit = qiskit.QuantumCircuit(_qr, _cr)
        quantum_circuit.name = name

        # initialize the system at the first place
        quantum_circuit.initialize(self._initial_state, 0)

        # add collisions and measurements
        for n in range(1, self._NCOLL + 1):
            quantum_circuit.append(
                qiskit.extensions.UnitaryGate(self._Utilde[n], label=r'$\tilde{U}_' + f'{n}$'),
                qargs=[0, self._NCOLL + 1 - n]
            )
            quantum_circuit.measure(qubit=self._NCOLL + 1 - n, cbit=n - 1)

        return quantum_circuit

    def set_simulation_parameters(self, NTRAJ: int = 20000, method: str = 'simulator', backend: str = None):
        self._NTRAJ = NTRAJ
        self._method = method
        self._backend = backend

    def _get_simulated_trajectories(self, experiment_name: str = None) -> np.ndarray:
        if experiment_name is None:
            experiment_name = 'temp'

        return simulate_quantum_circuits(qc_stack=[self.get_doobed_quantum_circuit(experiment_name)],
                                         NTRAJ=self._NTRAJ, how=self._method,
                                         backend_name=self._backend)[experiment_name].astype(str)

    def analyse_full_counting_statistics(self, experiment_name: str = None, num_resamples: int = 1) -> Union[
        pd.Series, list[pd.Series]]:

        # prepare data for analysis
        trajs_str = self._get_simulated_trajectories(experiment_name=experiment_name)
        df = pd.DataFrame(trajs_str, columns=['trajs.'])

        if df.size != self._NTRAJ:
            print('Attention! NTRAJ is different from data-size. Size of data is new NTRAJ.')
            self._NTRAJ = df.size

        # count occurrences and calculate probabilities
        fcs_hist = (df.groupby('trajs.').size() / self._NTRAJ).to_frame('P(k)')

        fcs_hist = self._reindex_fcs(fcs_hist)

        # do statistical analysis via resampling
        if num_resamples > 1:
            hists_resampled = pd.DataFrame(index=fcs_hist.index)
            num_per_resample = int(self._NTRAJ / num_resamples)
            # calculate probabilities for subsamples
            for i in range(num_resamples):
                sub_df = df.iloc[i * num_per_resample: (i + 1) * num_per_resample].groupby(
                    'trajs.').size() / self._NTRAJ * num_resamples
                hists_resampled[str(i)] = sub_df

            # calculate errors as range from biggest to smallest probability for each traj. (different % possible
            upper_err = np.percentile(hists_resampled.fillna(0), 100, axis=1) - fcs_hist
            lower_err = fcs_hist - np.percentile(hists_resampled.fillna(0), 0, axis=1)

            return [fcs_hist, lower_err, upper_err]

        else:
            return fcs_hist

    def _reindex_fcs(self, fcs_hist: pd.DataFrame):
        # rearrange index according to observable value
        fcs_hist['obs_val'] = [calculate_observable(k, self._biasing_pattern) for k in fcs_hist.index]
        fcs_hist.sort_values('obs_val', inplace=True)

        return fcs_hist['P(k)']

    def analyse_observable_histogram(self, experiment_name: str = None, num_resamples: int = 1) -> Union[
        np.ndarray, list[np.ndarray]]:

        # prepare data for analysis
        trajs_str = self._get_simulated_trajectories(experiment_name=experiment_name)

        if trajs_str.size != self._NTRAJ:
            print('Attention! NTRAJ is different from data-size. Size of data is new NTRAJ.')
            self._NTRAJ = trajs_str.size

        # calculate observable in occupational basis
        data = np.array([[int(s) for s in traj_str] for traj_str in trajs_str])
        data = data * self._biasing_pattern

        # calculate O(traj.) and make histogram
        obs_results = data.sum(axis=1)
        obs_hist = np.histogram(obs_results, bins=self._get_bins_for_obs_hist())[0]
        obs_hist = obs_hist / obs_hist.sum()

        # do statistical analysis
        if num_resamples > 1:
            hists_resampled = np.zeros((num_resamples, self.get_possible_observable_values().size))
            obs_reshaphed = obs_results.reshape(num_resamples, int(obs_results.size / num_resamples))
            for i in range(hists_resampled.shape[0]):
                hist_helper = np.histogram(obs_reshaphed[i, :], bins=self._get_bins_for_obs_hist())[0]
                hists_resampled[i, :] = hist_helper / hist_helper.sum()

            # calculate errors as range from biggest to smallest probability for each traj. (different % possible
            upper_err = np.percentile(hists_resampled, 100, axis=0) - obs_hist
            lower_err = obs_hist - np.percentile(hists_resampled, 0, axis=0)

            return [obs_hist, lower_err, upper_err]

        else:
            return obs_hist

    def analyse_temporal_outcomes(self, experiment_name: str = None, num_resamples: int = 1) -> Union[
        np.ndarray, list[np.ndarray]]:

        # prepare data for analysis
        trajs_str = self._get_simulated_trajectories(experiment_name=experiment_name)
        trajs = np.array([[int(s) for s in traj_str] for traj_str in trajs_str])

        if trajs_str.size != self._NTRAJ:
            print('Attention! NTRAJ is different from data-size. Size of data is new NTRAJ.')
            self._NTRAJ = trajs_str.size

        # calculate probabilities for '1' outcome
        data_01 = trajs.sum(axis=0) / self._NTRAJ

        size_new_resample = int(self._NTRAJ / num_resamples)
        if num_resamples > 1:
            results_resampled = np.zeros((num_resamples, self._NCOLL))
            for i in range(results_resampled.shape[0]):
                results_resampled[i, :] = trajs[size_new_resample * i:size_new_resample * (i + 1), :].sum(
                    axis=0) / size_new_resample

            upper_err = np.percentile(results_resampled, 100, axis=0) - data_01
            lower_err = data_01 - np.percentile(results_resampled, 0, axis=0)

            return [data_01, lower_err, upper_err]

        else:
            return data_01
