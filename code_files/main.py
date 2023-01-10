import matplotlib.pyplot as plt

from src import *


def make_overview_subplots():
    # plot full trajectory to later generate 'arbitrary' trajectories:
    # define number of collisions and plot vertical lines for "1" outcomes
    NCOLL = 4
    fig, ax = plt.subplots(figsize=(1.662, 0.5))
    for traj_ind in range(1, NCOLL + 1):
        ax.vlines(traj_ind, 0, 1, 'b')

    ax.set_xticks(np.arange(1, NCOLL + 1))
    ax.set_xticklabels([])
    ax.set_yticks([1])
    ax.set_ylim([0, 1.4])
    ax.set_xlim([0.5, NCOLL + 0.5])
    ax.tick_params(axis='both', which='both', direction='inout')

    fig.tight_layout()
    fig.savefig("figures/full_trajectories.svg", bbox_inches='tight', transparent=True)
    del fig, ax

    # create biased and original statistics:
    # define original collision model to calculate K_k
    DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=magnetization_pattern(NCOLL=NCOLL),
                                s_val=0)
    for i, s in enumerate([0, -4]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.8, 0.9))
        # calculate the full counting statistics P(\vec{k}) with/without reweighting
        fcs = calculate_full_counting_statistics(Kraus_list=DoobObj.get_undoobed_Kraus_ops(),
                                                 biasing_pattern=magnetization_pattern(NCOLL=NCOLL), s_val=s)
        # and analyse P(k_n=1)
        tpd = analyse_temporal_probability(full_counting_statistics=fcs)
        ax.fill_between(np.arange(1, NCOLL + 1), tpd, color={'b' if i == 1 else 'k'}, alpha=0.4)

        ax.set_xticks(np.arange(1, NCOLL + 1))
        ax.set_xticklabels([r'$k_{' + str(ind) + r'}$' for ind in range(1, 5)])
        ax.set_yticks([1])
        ax.set_ylim([0, 1.8])
        ax.set_xlim([0.28, NCOLL + 0.72])

        ax.tick_params(axis='both', which='both', direction='inout')

        fig.tight_layout()
        fig.savefig(f"figures/{'biased' if i == 1 else 'original'}_probabilities.svg", bbox_inches='tight',
                    transparent=True)

    plt.show()


def make_classical_sim_results(data_filename: str = None, if_new_then: str = 'classical'):
    fig = plt.figure(figsize=(3.3, 2.5), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    axs = [ax0, ax1, ax2]

    # prepare for DoobObj with respect to uniform and staggered field
    NCOLL = 4
    s_vals = np.linspace(-2, 2, 3)
    biasing_patterns = np.array(
        [magnetization_pattern(NCOLL=NCOLL),
         imbalance_pattern(NCOLL=NCOLL)]
    )
    qc_to_simulate = []

    for s_ind in range(s_vals.size):
        for p_ind in range(biasing_patterns.shape[0]):
            # calculate operators for biased dynamics
            DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=biasing_patterns[p_ind],
                                        s_val=s_vals[s_ind])
            # exact probabilities of \vec{k} and \vec{p}*\vec{k}
            fcs = calculate_full_counting_statistics(Kraus_list=DoobObj.get_undoobed_Kraus_ops(),
                                                     biasing_pattern=biasing_patterns[p_ind], s_val=s_vals[s_ind])
            opd = analyse_observable_histogram(full_counting_statistics=fcs, biasing_pattern=biasing_patterns[p_ind])

            axs[p_ind].fill_between(opd.index,
                                    opd['$P_s(O)$'],
                                    color=__line_colors[s_ind], alpha=.4)

            # prepare circuit for later simulation
            qc_to_simulate.append(
                DoobObj.get_doobed_quantum_circuit(name=f's={s_vals[s_ind]:.0f}_and_pattern_no.{p_ind}'))

            axs[p_ind].set_xticks(DoobObj.get_possible_observable_values())

    # add full counting trajectories for 'arbitrary' field to last plot
    NCOLL = 3
    custom_pattern = -np.ones(NCOLL)
    custom_pattern[2::3] = 1

    for s_ind in range(s_vals.size):
        DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=custom_pattern,
                                    s_val=s_vals[s_ind])
        fcs = calculate_full_counting_statistics(Kraus_list=DoobObj.get_undoobed_Kraus_ops(),
                                                 biasing_pattern=custom_pattern,
                                                 s_val=s_vals[s_ind])

        axs[2].fill_between(np.arange(fcs.index.size),
                            fcs,
                            color=__line_colors[s_ind], alpha=0.4, label=f'$s={s_vals[s_ind]:.0f}$')

        axs[2].set_xticks(np.arange(fcs.index.size), fcs.index, rotation=45)

        # prepare for sim
        qc_to_simulate.append(
            DoobObj.get_doobed_quantum_circuit(name=f's={s_vals[s_ind]:.0f}_and_pattern_no.2'))

    # make simulation if necessary
    if data_filename is None:
        method = 'ibmq' if if_new_then == 'quantum' else 'simulator'
        simulate_quantum_circuits(qc_stack=qc_to_simulate, NTRAJ=__NTRAJ, how=method, backend_name='ibm_oslo')

        #  get simulations name
        with open('src/number_of_' + if_new_then + '_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            preceding_number = ''
            for _ in range(6 - len(list(str(number - 1)))):
                preceding_number += '0'
            preceding_number += str(number - 1)

        data_filename = if_new_then + '_sim_' + preceding_number
    # else: use given data_filename to retrieve already calculated data

    # visualize the simulation results
    for s_ind in range(s_vals.size):
        for p_ind in range(biasing_patterns.shape[0]):
            DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=biasing_patterns[p_ind],
                                        s_val=s_vals[s_ind])
            DoobObj.set_simulation_parameters(NTRAJ=__NTRAJ, method='from_hdf5_' + data_filename)
            sim_results = DoobObj.analyse_observable_histogram(
                experiment_name=f's={s_vals[s_ind]:.0f}_and_pattern_no.{p_ind}', num_resamples=20)

            axs[p_ind].errorbar(DoobObj.get_possible_observable_values(),
                                sim_results[0],  # [sim_results[1], sim_results[2]],
                                fmt=__marker[if_new_then],
                                c=__line_colors[s_ind],
                                markersize=4)

    for s_ind in range(s_vals.size):
        DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=custom_pattern,
                                    s_val=s_vals[s_ind])
        DoobObj.set_simulation_parameters(NTRAJ=__NTRAJ, method='from_hdf5_' + data_filename)
        sim_results = DoobObj.analyse_full_counting_statistics(
            experiment_name=f's={s_vals[s_ind]:.0f}_and_pattern_no.2', num_resamples=20)

        axs[2].errorbar(np.arange(sim_results[0].index.size),
                        sim_results[0],  # [sim_results[1], sim_results[2]],
                        fmt=__marker[if_new_then],
                        c=__line_colors[s_ind], markersize=4)

    # make annotations
    titles = [r'(a) Uniform field', '(b) Staggered field', "(c) General field"]
    # x_labels = [r'$\mathcal{O}_\mathbf{p}(\mathbf{k}) = \mathbf{p}\cdot \mathbf{k}$',
    #             r'$\mathcal{O}_\mathbf{p}(\mathbf{k}) = \mathbf{p}\cdot \mathbf{k}$', '$\mathbf{k}$']
    y_labels = ['$P$', '', '$P$']
    annotation = [r'$\mathbf{p} = (1, 1, 1, 1)$', r'$\mathbf{p} = (-1, 1, -1, 1)$', r'$\mathbf{p} = (-1, -1, 1)$']

    xticks = ['(' + str(k[0]) + ',' + str(k[1]) + ',' + str(k[2]) + ')' for k in sim_results[0].index]
    axs[2].set_xticks(np.arange(len(xticks)), xticks)

    for i in range(3):
        axs[i].set_title(titles[i], fontsize=10)
        # axs[i].set_xlabel(x_labels[i])
        axs[i].set_ylabel(y_labels[i])
        axs[i].annotate(annotation[i], xy=(.1, .8), xycoords='axes fraction')

        axs[i].tick_params(axis='both', which='both', direction='inout')

    axs[1].set_yticklabels([])
    # axs[2].legend()

    for i in range(2):
        axs[i].set_ylim([0, 0.9])
    axs[2].set_ylim([0, 0.75])

    fig.savefig(f'figures/fig_2_from_{data_filename}.svg', bbox_inches='tight',
                transparent=False)
    plt.show()


def make_performance_test_of_quantum_computer(data_filename: str = None, if_new_then: str = 'classical'):
    fig = plt.figure(figsize=(3.3, 2.5), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[0, 2:])
    ax1 = fig.add_subplot(gs[1, :])
    axs = [ax0, ax1]

    # prepare uniform field, now with intention to test it on QPC
    NCOLL = 3
    s_vals = np.linspace(-2, 2, 3)
    biasing_pattern = magnetization_pattern(NCOLL=NCOLL)

    qc_to_simulate = []

    # preparation and calculations similar to fig. 2
    for s_ind in range(s_vals.size):
        DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=biasing_pattern,
                                    s_val=s_vals[s_ind])
        # exact probabilities
        fcs = calculate_full_counting_statistics(Kraus_list=DoobObj.get_undoobed_Kraus_ops(),
                                                 biasing_pattern=biasing_pattern, s_val=s_vals[s_ind])
        opd = analyse_observable_histogram(full_counting_statistics=fcs, biasing_pattern=biasing_pattern)

        # plot probabilities of magnetization
        axs[0].fill_between(opd.index,
                            opd['$P_s(O)$'],
                            color=__line_colors[s_ind], alpha=.4)

        # plot full counting statistics
        axs[1].fill_between(np.arange(fcs.index.size),
                            fcs,
                            color=__line_colors[s_ind], alpha=0.4, label=f'$s={s_vals[s_ind]:.0f}$')

        axs[1].set_xticks(np.arange(fcs.index.size), fcs.index, rotation=45)

        # prepare for sim
        qc_to_simulate.append(
            DoobObj.get_doobed_quantum_circuit(name=f's={s_vals[s_ind]:.0f}'))

    # make simulation if necessary
    if data_filename is None:
        method = 'ibmq' if if_new_then == 'quantum' else 'simulator'
        simulate_quantum_circuits(qc_stack=qc_to_simulate, NTRAJ=__NTRAJ, how=method, backend_name='ibm_oslo')

        #  get simulations name
        with open('src/number_of_' + if_new_then + '_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            preceding_number = ''
            for _ in range(6 - len(list(str(number - 1)))):
                preceding_number += '0'
            preceding_number += str(number - 1)

        data_filename = if_new_then + '_sim_' + preceding_number

    for s_ind in range(s_vals.size):
        DoobObj = TimeDependentDoob(collision_unitary=unbiased_unitary, biasing_pattern=biasing_pattern,
                                    s_val=s_vals[s_ind])
        DoobObj.set_simulation_parameters(NTRAJ=__NTRAJ, method='from_hdf5_' + data_filename)

        # magnetization
        sim_results = DoobObj.analyse_observable_histogram(
            experiment_name=f's={s_vals[s_ind]:.0f}', num_resamples=20)

        axs[0].errorbar(DoobObj.get_possible_observable_values(),
                        sim_results[0],  # [sim_results[1], sim_results[2]],
                        marker=__marker[if_new_then], linestyle='None',
                        c=__line_colors[s_ind], markersize=4)

        # full counting statistics
        sim_results = DoobObj.analyse_full_counting_statistics(
            experiment_name=f's={s_vals[s_ind]:.0f}', num_resamples=20)

        # plot full counting statistics
        axs[1].errorbar(np.arange(sim_results[0].index.size),
                        sim_results[0],  # [sim_results[1], sim_results[2]],
                        marker=__marker[if_new_then], linestyle='None',
                        c=__line_colors[s_ind], markersize=4)

        axs[0].set_xticks(DoobObj.get_possible_observable_values())

    # make annotations
    titles = ['(b) Occupation', '(c) Individual trajectories']
    # x_labels = [r'$\mathcal{O}_\mathbf{p}(\mathbf{k}) = \mathbf{p}\cdot \mathbf{k}$', '$\mathbf{k}$']
    y_labels = ['$P_s$', '$P_s$']

    xticks = ['(' + str(k[0]) + ',' + str(k[1]) + ',' + str(k[2]) + ')' for k in sim_results[0].index]
    axs[1].set_xticks(np.arange(len(xticks)), xticks)

    for i in range(2):
        axs[i].set_title(titles[i], fontsize=10)
        # axs[i].set_xlabel(x_labels[i])
        axs[i].set_ylabel(y_labels[i])
        axs[i].set_ylim([0, axs[i].get_ylim()[1]])

        axs[i].annotate(r'$\mathbf{p} = (1, 1, 1)$', xy=(.2 + 0.15 * i, .8), xycoords='axes fraction')

        axs[i].tick_params(axis='both', which='both', direction='inout')

    # axs[1].legend()
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()

    fig.savefig(f'figures/fig_3_from_{data_filename}.svg', bbox_inches='tight', transparent=False)

    plt.show()


def make_all_correlation_tests(data_filename_classical: str = None, data_file_name_quantum: str = None) -> None:
    fig = plt.figure(figsize=(3.3, 2.8), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(4, 4, figure=fig)
    ax0_cl = fig.add_subplot(gs[0, 2:])
    ax0_qt = fig.add_subplot(gs[1, 2:], sharex=ax0_cl, sharey=ax0_cl)
    ax1_cl = fig.add_subplot(gs[2, :])
    ax1_qt = fig.add_subplot(gs[3, :], sharex=ax1_cl, sharey=ax1_cl)
    plt.setp(ax0_cl.get_xticklabels(), visible=False)
    plt.setp(ax1_cl.get_xticklabels(), visible=False)

    # prepare list, that contains specifications of the nearest neighbour energy
    NCOLL = 3
    s_vals = np.linspace(-1, 1, 3)
    correlation_obs_spec = correlation_specs(NCOLL=NCOLL)

    qc_to_simulate = []

    # formally similar to fig. 2/3, but GeneralTimeDependentDoob-class has greater versatility
    # (especially biased dynamics for interacting energy-functions)
    for s_ind in range(s_vals.size):
        DoobObj = GeneralTimeDependentDoob(collision_unitary=unbiased_unitary, observable_specs=correlation_obs_spec,
                                           s_val=s_vals[s_ind])
        # exact probabilities
        fcs = calculate_full_counting_statistics_functional(Kraus_list=DoobObj.get_undoobed_Kraus_ops(),
                                                            observable_specs=correlation_obs_spec, s_val=s_vals[s_ind])
        opd = analyse_observable_histogram_functional(full_counting_statistics=fcs,
                                                      observable_specs=correlation_obs_spec)

        # plot probabilities of magnetization
        for ax in [ax0_cl, ax0_qt]:
            ax.fill_between(opd.index,
                            opd['$P_s(O)$'],
                            color=__line_colors[s_ind], alpha=.4)

        # plot full counting statistics
        for ax in [ax1_cl, ax1_qt]:
            ax.fill_between(np.arange(fcs.index.size),
                            fcs,
                            color=__line_colors[s_ind], alpha=0.4, label=f'$s={s_vals[s_ind]:.0f}$')

            ax.set_xticks(np.arange(fcs.index.size), fcs.index, rotation=45)

        # prepare for sim
        qc_to_simulate.append(
            DoobObj.get_doobed_quantum_circuit(name=f's={s_vals[s_ind]:.0f}'))

    # make simulation if necessary
    if data_filename_classical is None:
        simulate_quantum_circuits(qc_stack=qc_to_simulate, NTRAJ=__NTRAJ, how='simulator', backend_name='None')

        #  get simulations name
        with open('src/number_of_classical_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            preceding_number = ''
            for _ in range(6 - len(list(str(number - 1)))):
                preceding_number += '0'
            preceding_number += str(number - 1)

        data_filename_classical = 'classical_sim_' + preceding_number

    if data_file_name_quantum is None:
        simulate_quantum_circuits(qc_stack=qc_to_simulate, NTRAJ=__NTRAJ, how='ibmq', backend_name='ibm_oslo')

        #  get simulations name
        with open('src/number_of_quantum_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            preceding_number = ''
            for _ in range(6 - len(list(str(number - 1)))):
                preceding_number += '0'
            preceding_number += str(number - 1)

        data_file_name_quantum = 'classical_sim_' + preceding_number

    axs_list_dict = [{'classical': ax0_cl, 'quantum': ax0_qt}, {'classical': ax1_cl, 'quantum': ax1_qt}]

    for s_ind in range(s_vals.size):
        DoobObj = GeneralTimeDependentDoob(collision_unitary=unbiased_unitary, observable_specs=correlation_obs_spec,
                                           s_val=s_vals[s_ind])
        for method, file in zip(['classical', 'quantum'
                                 ], [data_filename_classical, data_file_name_quantum
                                     ]):
            DoobObj.set_simulation_parameters(NTRAJ=__NTRAJ, method='from_hdf5_' + file)

            sim_results = DoobObj.analyse_observable_histogram(
                experiment_name=f's={s_vals[s_ind]:.0f}', num_resamples=20)

            axs_list_dict[0][method].errorbar(DoobObj.get_possible_observable_values(),
                                              sim_results[0],  # [sim_results[1], sim_results[2]],
                                              marker=__marker[method], linestyle='None',
                                              c=__line_colors[s_ind], markersize=4)

            # full counting statistics
            sim_results = DoobObj.analyse_full_counting_statistics(
                experiment_name=f's={s_vals[s_ind]:.0f}', num_resamples=20)

            # plot full counting statistics
            axs_list_dict[1][method].errorbar(np.arange(sim_results[0].index.size),
                                              sim_results[0],  # [sim_results[1], sim_results[2]],
                                              marker=__marker[method], linestyle='None',
                                              c=__line_colors[s_ind], markersize=4)

        axs_list_dict[0][method].set_xticks(DoobObj.get_possible_observable_values())

    # make annotations
    titles = ['(b) Correlations', '(c) Individual trajectories']
    # x_labels = [r'$\mathcal{O}_\text{NN}(\mathbf{k})$', '$\mathbf{k}$']
    y_labels = ['$P_s$', '$P_s$']

    xticks = ['(' + str(k[0]) + ',' + str(k[1]) + ',' + str(k[2]) + ')' for k in sim_results[0].index]
    ax1_qt.set_xticks(np.arange(len(xticks)), xticks)

    for i, ax in enumerate([ax0_qt, ax1_qt]):
        # ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])
        ax.set_ylim([0, ax.get_ylim()[1]])

    for i, ax in enumerate([ax0_cl, ax1_cl]):
        ax.set_title(titles[i], fontsize=10)

    for ax in [ax0_cl, ax0_qt, ax1_cl, ax1_qt]:
        ax.tick_params(axis='both', which='both', direction='inout')

    # ax1_qt.legend()
    for ax in [ax0_cl, ax0_qt]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    fig.savefig(f'figures/fig_4_from_{data_file_name_quantum}.svg', bbox_inches='tight',
                transparent=False)

    plt.show()


# standard values
__omega = 1
__kappa = 1
__NTRAJ = 20000  # max. free shots on QPC
__line_colors = ['b', 'k', 'r']
__marker = {'classical': '+', 'quantum': r'$\circ$'}

# calculate collisions hamiltonian and unitary
unbiased_hamiltonian = collision_hamiltonian(omega=__omega, kappa=__kappa)
unbiased_unitary = collision_unitary(hamiltonian=unbiased_hamiltonian)

if __name__ == '__main__':
    make_overview_subplots()
    make_classical_sim_results(data_filename='classical_sim_000001', if_new_then='classical')
    make_performance_test_of_quantum_computer(data_filename='quantum_sim_000001', if_new_then='quantum')
    make_all_correlation_tests(data_filename_classical='classical_sim_000003',
                               data_file_name_quantum='quantum_sim_000002')

    plt.show()
