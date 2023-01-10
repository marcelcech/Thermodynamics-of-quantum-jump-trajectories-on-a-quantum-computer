from .package_requirements import *
from .two_level_system_helper import ket1, ket0
from .TimeDependentDoob import TimeDependentDoob
from .special_observables import magnetization_pattern
from .Stinespring_helper import find_orthogonal_vec
from .exact_calculations_general_version import calculate_observable_functional


class GeneralTimeDependentDoob(TimeDependentDoob):

    def __init__(self, collision_unitary: qt.Qobj, observable_specs: list, s_val: float):
        self._observable_specs = observable_specs
        self._depth = observable_specs[1]
        self._obs_func = observable_specs[2]

        super().__init__(collision_unitary=collision_unitary,
                         biasing_pattern=magnetization_pattern(NCOLL=observable_specs[0]), s_val=s_val)
        K_ops = self.get_undoobed_Kraus_ops()
        K_super = [qt.sprepost(K_ops[i], K_ops[i].dag()) for i in range(2)]

        list_helper = [qt.qeye(2) for _ in range(2)]
        for ind in range(self._depth - 2):
            list_helper = deepcopy([deepcopy(list_helper) for _ in range(2)])

        _G_n_list = [deepcopy(list_helper) for _ in range(self._NCOLL + 1)]
        _inv_G_n_list = deepcopy(_G_n_list)

        for n in np.arange(self._NCOLL, 0, -1):
            if n > self._depth - 1:
                factor = np.exp(-self._s_val)
            else:
                factor = 1

            for index_nm1 in range(max(2, 2 ** (self._depth - 1))):
                dummy_pointer_nm1 = _G_n_list[n - 1]
                dummy_inv_pointer_nm1 = _inv_G_n_list[n - 1]

                bin_index_nm1 = "{0:b}".format(index_nm1)
                bin_index_nm1 = ["0"] * (self._depth - 1 - len(bin_index_nm1)) + list(bin_index_nm1)
                bin_index_nm1 = [int(letter) for letter in bin_index_nm1]

                for ind in bin_index_nm1[:-1]:
                    dummy_pointer_nm1 = dummy_pointer_nm1[ind]
                    dummy_inv_pointer_nm1 = dummy_inv_pointer_nm1[ind]

                sum_helper = qt.Qobj(np.zeros((2, 2)))
                for k in range(2):
                    pointer_n = _G_n_list[n]
                    bin_index_n = [k] + bin_index_nm1[:-1]
                    for ind in bin_index_n:
                        pointer_n = pointer_n[ind]

                    list_to_eval = [k] + bin_index_nm1[:self._depth - 1]
                    exponent = self._obs_func[(n - self._depth) % len(self._obs_func)](list_to_eval[::-1])

                    sum_helper += factor ** exponent * (
                        K_super[k].dag()(pointer_n ** 2)
                    )
                    assert (pointer_n - reduce(operator.getitem, bin_index_n, _G_n_list[n])).full().max() < 1e-9

                dummy_pointer_nm1[bin_index_nm1[-1]] = sum_helper.sqrtm()
                dummy_inv_pointer_nm1[bin_index_nm1[-1]] = sum_helper.sqrtm().inv()

                assert np.abs((dummy_pointer_nm1[bin_index_nm1[-1]] * dummy_inv_pointer_nm1[
                    bin_index_nm1[-1]] - qt.qeye(2)).full()).max() < 1e-9

        # insert some small tests
        for n in range(self._NCOLL, 0, -1):
            if n > self._depth - 1:
                factor = np.exp(-self._s_val)
            else:
                factor = 1

            for index_nm1 in range(2 ** (self._depth - 1)):
                bin_index_nm1 = "{0:b}".format(index_nm1)
                bin_index_nm1 = ["0"] * (self._depth - 1 - len(bin_index_nm1)) + list(bin_index_nm1)
                bin_index_nm1 = [int(letter) for letter in bin_index_nm1]

                qt_helper = qt.qzero(2)
                for k in range(2):
                    list_to_eval = [k] + bin_index_nm1[:self._depth - 1]
                    exponent = self._obs_func[(n - self._depth) % len(self._obs_func)](list_to_eval[::-1])

                    qt_helper += factor ** exponent * \
                                 K_super[k].dag()(
                                     reduce(operator.getitem, [k] + bin_index_nm1[:-1], _G_n_list[n]) ** 2
                                 )

                assert np.abs((qt_helper - reduce(operator.getitem, bin_index_nm1,
                                                  _G_n_list[n - 1]) ** 2).full()).max() < 1e-9

        list_helper = []
        for _ in range(self._depth - 2):
            list_helper = [deepcopy(list_helper) for _ in range(2)]

        self._Ktilde_n = [deepcopy(list_helper) for _ in range(self._NCOLL + 1)]

        for n in range(1, self._NCOLL + 1):
            if n > self._depth - 1:
                factor = np.exp(-self._s_val / 2)
            else:
                factor = 1

            pointer_K_n = self._Ktilde_n[n]
            for index_nm1 in range(2 ** (self._depth - 1)):
                dummy_pointer_K_n = pointer_K_n

                bin_index_nm1 = "{0:b}".format(index_nm1)
                bin_index_nm1 = ["0"] * (self._depth - 1 - len(bin_index_nm1)) + list(bin_index_nm1)
                bin_index_nm1 = [int(letter) for letter in bin_index_nm1]

                for ind in bin_index_nm1[:-1]:
                    dummy_pointer_K_n = dummy_pointer_K_n[ind]

                qt_helper = [qt.qzero(2) for _ in range(2)]
                for k in range(2):
                    list_to_eval = [k] + bin_index_nm1[:self._depth - 1]
                    exponent = self._obs_func[(n - self._depth) % len(self._obs_func)](list_to_eval[::-1])

                    qt_helper[k] = factor ** exponent * (
                            reduce(operator.getitem, [k] + bin_index_nm1[:-1], _G_n_list[n]) * K_ops[k] *
                            reduce(operator.getitem, bin_index_nm1, _inv_G_n_list[n - 1])
                    )

                assert np.abs((qt_helper[0].dag() * qt_helper[0] + qt_helper[1].dag() * qt_helper[1] - qt.qeye(
                    2)).full()).max() < 1e-9

                dummy_pointer_K_n.append(qt_helper)

        # calculate new initial state
        null_index = "{0:b}".format(0)
        null_index = ["0"] * (self._depth - 1 - len(null_index)) + list(null_index)
        null_index = [int(letter) for letter in null_index]

        _G_0 = reduce(operator.getitem, null_index, _G_n_list[0])
        _psi_tilde_0 = (_G_0 * ket0).unit()
        self._initial_state = [x[0] for x in _psi_tilde_0.full().tolist()]

        self._Utilde = [None]
        # first collision
        Ktilde_ops = reduce(operator.getitem, null_index, self._Ktilde_n[1])
        kets = [ket0, ket1]
        U_helper = np.zeros((4, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_helper[2 * i + j, k] = (kets[j].dag() * Ktilde_ops[i] * kets[k]).full()[0][0]

        for i in range(2):
            U_helper = np.append(U_helper, find_orthogonal_vec(U_helper).reshape((4, 1)), axis=1)
            U_helper[:, 2 + i] = U_helper[:, 2 + i] / np.linalg.norm(U_helper[:, 2 + i])
            # U_helper = np.append(U_helper, U_helper[::-1, -1 - 2 * i].reshape((4, 1)), axis=1)

        assert np.abs((qt.Qobj(U_helper) * qt.Qobj(U_helper).dag() - qt.qeye(4)).full()).max() < 1e-10
        self._Utilde.append(U_helper)

        # and the rest
        for n in range(2, self._NCOLL + 1):
            _Utilde_helper = qt.Qobj()
            for index_nm1 in range(2 ** min(n - 1, self._depth - 1)):
                bin_index_nm1 = "{0:b}".format(index_nm1)
                bin_index_nm1 = ["0"] * (self._depth - 1 - len(bin_index_nm1)) + list(bin_index_nm1)
                bin_index_nm1 = [int(letter) for letter in bin_index_nm1]

                Ktilde_ops = reduce(operator.getitem, bin_index_nm1, self._Ktilde_n[n])
                kets = [ket0, ket1]
                U_helper = np.zeros((4, 2), dtype=complex)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            U_helper[2 * i + j, k] = (kets[j].dag() * Ktilde_ops[i] * kets[k]).full()[0][0]

                for i in range(2):
                    U_helper = np.append(U_helper, find_orthogonal_vec(U_helper).reshape((4, 1)), axis=1)
                    U_helper[:, 2 + i] = U_helper[:, 2 + i] / np.linalg.norm(U_helper[:, 2 + i])
                    # U_helper = np.append(U_helper, U_helper[::-1, -1 - 2 * i].reshape((4, 1)), axis=1)

                assert np.abs((qt.Qobj(U_helper) * qt.Qobj(U_helper).dag() - qt.qeye(4)).full()).max() < 1e-10

                _Utilde_helper += qt.tensor(
                    *[kets[ind].proj() for ind in bin_index_nm1[::-1][:min(n - 1, self._depth - 1)]],
                    qt.Qobj(U_helper))

            test_qeye = qt.tensor(*[qt.qeye(2) for _ in range(min(n - 1, self._depth - 1))], qt.qeye(4))
            assert np.abs((_Utilde_helper.dag() * _Utilde_helper - test_qeye).full()).max() < 1e-9

            self._Utilde.append(_Utilde_helper.full())

    def get_possible_observable_values(self) -> np.ndarray:
        _possible_small_trajs = [[]]
        for _ in range(self._depth):
            _copy_possible_small_trajs = copy(_possible_small_trajs)
            _possible_small_trajs = []
            for p_s_t in _copy_possible_small_trajs:
                for i in range(2):
                    p_s_t.append(i)
                    _possible_small_trajs.append(copy(p_s_t))
                    p_s_t = p_s_t[:-1]

        _possible_vals = [0]
        for n in range(self._depth, self._NCOLL + 1):
            _copy_possible_vals = copy(_possible_vals)
            _possible_vals = []
            for current_val in _copy_possible_vals:
                for p_s_t in _possible_small_trajs:
                    _possible_vals.append(current_val + self._obs_func[(n - self._depth) % len(self._obs_func)](p_s_t))
            _possible_vals = np.unique(_possible_vals)

        return _possible_vals

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
                qargs=[0, *list(range(self._NCOLL + 1 - n, min(self._NCOLL + 1 - n + self._depth, self._NCOLL + 1)))]
                # right?
            )
            quantum_circuit.measure(qubit=self._NCOLL + 1 - n, cbit=n - 1)

        return quantum_circuit

    def analyse_observable_histogram(self, experiment_name: str = None, num_resamples: int = 1) -> Union[
        np.ndarray, list[np.ndarray]]:

        # prepare data for analysis
        trajs_str = self._get_simulated_trajectories(experiment_name=experiment_name)

        if trajs_str.size != self._NTRAJ:
            print('Attention! NTRAJ is different from data-size. Size of data is new NTRAJ.')
            self._NTRAJ = trajs_str.size

        # calculate observable in occupational basis
        data = np.array([[int(s) for s in traj_str] for traj_str in trajs_str])

        # calculate O(traj.) and make histogram
        obs_result_helper = np.zeros((self._NTRAJ, self._NCOLL - self._depth + 1))
        for traj_ind in range(self._NTRAJ):
            for val_ind in range(self._depth, self._NCOLL + 1):
                obs_result_helper[traj_ind, val_ind - self._depth] = self._obs_func[
                    (val_ind - self._depth) % len(self._obs_func)](
                    data[traj_ind, val_ind - self._depth:val_ind].tolist()
                )

        obs_results = obs_result_helper.sum(axis=1)
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

    def _reindex_fcs(self, fcs_hist: pd.DataFrame):
        # rearrange index according to observable value
        fcs_hist['obs_val'] = [calculate_observable_functional(k, self._observable_specs) for k in fcs_hist.index]
        fcs_hist.sort_values('obs_val', inplace=True)

        return fcs_hist['P(k)']
