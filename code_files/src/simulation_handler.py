import numpy as np

from .package_requirements import *
from .hdf5_handler import read_hdf5_to_dict, to_hdf5_from_dict


# after : qiskit.IBMQ.save_account(...)
def ibmq_login() -> None:
    qiskit.IBMQ.load_account()


def simulate_quantum_circuits(qc_stack: list[qiskit.QuantumCircuit], NTRAJ: int, how: str = 'simulator',
                              backend_name: str = 'None', additional_info: str = "None") -> dict:
    # return already calculated data
    if 'from_hdf5_' in how:
        return read_hdf5_to_dict(f'./data/{how[10:]}.h5')

    if 'retrieve_' in how:
        ibmq_login()
        provider = qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(backend_name)

        results = backend.retrieve_job(how[9:]).result()

        with open('src/number_of_quantum_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            next_number = ''
            for _ in range(6 - len(list(str(number + 1)))):
                next_number += '0'
            next_number += str(number + 1)

        counter_file = open('src/number_of_quantum_simulations.txt', 'w')
        counter_file.write(next_number)
        counter_file.close()

        job_name = 'quantum_sim_' + current_sim_number

    # or calculate new data
    elif how == 'simulator':
        sim = qiskit.Aer.get_backend('aer_simulator')  # for simulation of an ideal qpc; otherwise qasm_simulator
        results = sim.run(qc_stack, memory=True, shots=NTRAJ).result()

        with open('src/number_of_classical_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            next_number = ''
            for _ in range(6 - len(list(str(number + 1)))):
                next_number += '0'
            next_number += str(number + 1)

        counter_file = open('src/number_of_classical_simulations.txt', 'w')
        counter_file.write(next_number)
        counter_file.close()

        job_name = 'classical_sim_' + current_sim_number


    elif how == 'ibmq':
        ibmq_login()
        provider = qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(backend_name)

        transpiled_qc = qiskit.transpile(qc_stack, backend=backend)

        job = backend.run(transpiled_qc, memory=True, shots=NTRAJ)

        with open('src/number_of_quantum_simulations.txt', 'r+') as file:
            current_sim_number = file.read()
            number = int(current_sim_number)

            next_number = ''
            for _ in range(6 - len(list(str(number + 1)))):
                next_number += '0'
            next_number += str(number + 1)

        counter_file = open('src/number_of_quantum_simulations.txt', 'w')
        counter_file.write(next_number)
        counter_file.close()

        job_name = 'quantum_sim_' + current_sim_number

        print('Job-ID:', job.job_id())
        qiskit.tools.job_monitor(job)

        results = job.result()

    else:
        print(f"This simulator {how} is not available.")
        results = None
        job_name = None
        quit(-1)

    # save it to hdf5 for later
    experiments_list = [experiment['header']['name'] for experiment in results.to_dict()['results']]

    dict_to_save = {
        experiment_name: np.array(
            [string[::-1] for string in _get_single_result_memory(results, experiment_name=experiment_name)],
            dtype=h5str)
        for experiment_name in experiments_list
    }

    dict_to_save.update({'general_info': np.array(['info in parameters'], dtype=h5str),
                         'general_info_parameters': {'additional info': additional_info}})

    to_hdf5_from_dict(f'./data/{job_name}.h5', dict_to_save)

    return dict_to_save


def read_data_from_dict(data_dict: dict, experiment_name: str) -> np.ndarray:
    return data_dict[experiment_name].astype(str)


def _get_single_result_memory(results: Union[qiskit.result.Result, dict], experiment_name: str = None) -> list[str]:
    if experiment_name is None:
        return results.get_memory()

    else:
        return results.get_memory(experiment_name)
