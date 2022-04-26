import itertools

from typing import List, Union


class Experiment:
    def __init__(self,
                 identifier: str,
                 executable: str,
                 arguments: List[Union[float, int, str]],
                 columns: List[str],
                 repetitions: int = 5,
                 profilers: List[str] = None):

        self.identifier = identifier
        self.executable = executable
        self.arguments = arguments
        self.columns = columns
        self.repetitions = repetitions
        self.profilers = profilers


def InitExperiments():

    global experiments_path
    experiments_path = "../experiments"

    global experiments
    experiments = []

    ####################################################################################################################
    # radix-mgpu-sort
    ####################################################################################################################

    executable = "radix-mgpu-sort"
    columns = [
        "cuda_device_name", "num_system_gpus", "num_keys", "gpus",
        "num_cpu_threads", "data_type", "distribution_type",
        "distribution_parameter", "chunk_size", "num_thread_blocks",
        "device_memory_in_bytes", "device_memory_overhead",
        "local_sort_memory_overhead", "num_partitioning_passes_needed",
        "sort_duration_breakdown", "total_sort_duration"
    ]
    output_mode = ["csv"]

    ####################################################################################################################
    # num_keys_to_sort_duration
    ####################################################################################################################

    identifier = "num_keys_to_sort_duration"
    arguments = []

    data_type = ["uint32"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    num_keys = [1000, 1000000, 1000000000, 2000000000, 4000000000]
    gpus = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [
        1000, 1000000, 1000000000, 2000000000, 4000000000, 6000000000,
        8000000000
    ]
    gpus = ["0,1", "0,2"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [
        1000, 1000000, 1000000000, 2000000000, 4000000000, 6000000000,
        8000000000, 10000000000, 12000000000
    ]
    gpus = ["0,1,2", "0,2,4"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [
        1000, 1000000, 1000000000, 2000000000, 4000000000, 6000000000,
        8000000000, 10000000000, 12000000000, 14000000000, 16000000000
    ]
    gpus = ["0,1,2,3", "0,2,4,6"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [
        1000, 1000000, 1000000000, 2000000000, 4000000000, 6000000000,
        8000000000, 10000000000, 12000000000, 14000000000, 16000000000,
        18000000000, 20000000000, 22000000000, 24000000000, 26000000000,
        28000000000, 30000000000, 32000000000, 34000000000
    ]
    gpus = ["0,1,2,3,4,5,6,7"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(Experiment(identifier, executable, arguments, columns))

    ####################################################################################################################
    # data_type_to_sort_duration
    ####################################################################################################################

    identifier = "data_type_to_sort_duration"
    arguments = []

    num_keys = [2000000000, 4000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32", "float32"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [2000000000, 4000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32", "float32"]
    distribution_type = ["zipf"]
    distribution_parameter = ["1.0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [1000000000, 2000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint64", "float64"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [1000000000, 2000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint64", "float64"]
    distribution_type = ["zipf"]
    distribution_parameter = ["1.0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(Experiment(identifier, executable, arguments, columns))

    ####################################################################################################################
    # distribution_type_to_sort_duration
    ####################################################################################################################

    identifier = "distribution_type_to_sort_duration"
    arguments = []

    num_keys = [2000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = [
        "uniform",
        "normal",
        "zero",
        "sorted",
        "reverse-sorted",
        "nearly-sorted",
    ]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [2000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = ["skewed"]
    distribution_parameter = []
    for bit_entropy in range(33):
        distribution_parameter.append(bit_entropy)

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [2000000000]
    gpus = ["0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = ["zipf"]
    distribution_parameter = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        4.0
    ]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(Experiment(identifier, executable, arguments, columns))

    ####################################################################################################################
    # num_keys_to_sort_duration_profile_nsys
    ####################################################################################################################

    identifier = "num_keys_to_sort_duration_profile_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_keys = [2000000000, 4000000000]
    gpus = ["0", "0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32", "float32"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [1000000000, 2000000000]
    gpus = ["0", "0,1", "0,2", "0,1,2,3", "0,2,4,6", "0,1,2,3,4,5,6,7"]
    data_type = ["uint64", "float64"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(
        Experiment(identifier,
                   executable,
                   arguments,
                   columns,
                   repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # distribution_type_to_sort_duration_profile_nsys
    ####################################################################################################################

    identifier = "distribution_type_to_sort_duration_profile_nsys"
    repetitions = 1
    profilers = ["nsys"]
    arguments = []

    num_keys = [2000000000]
    gpus = ["0,1", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = [
        "uniform",
        "normal",
        "zero",
        "sorted",
        "reverse-sorted",
        "nearly-sorted",
    ]
    distribution_parameter = ["0"]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [2000000000]
    gpus = ["0,1", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = ["skewed"]
    distribution_parameter = []
    for bit_entropy in range(33):
        distribution_parameter.append(bit_entropy)

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    num_keys = [2000000000]
    gpus = ["0,1", "0,1,2,3,4,5,6,7"]
    data_type = ["uint32"]
    distribution_type = ["zipf"]
    distribution_parameter = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0
    ]

    arguments += list(
        itertools.product(*[
            num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(
        Experiment(identifier,
                   executable,
                   arguments,
                   columns,
                   repetitions=repetitions,
                   profilers=profilers))

    ####################################################################################################################
    # sort-benchmark
    ####################################################################################################################

    executable = "sort-benchmark"
    columns = [
        "cuda_device_name", "num_system_gpus", "sorting_algorithm", "num_keys",
        "gpus", "num_cpu_threads", "data_type", "distribution_type",
        "distribution_parameter", "chunk_size", "num_thread_blocks",
        "device_memory_in_bytes", "device_memory_overhead",
        "local_sort_memory_overhead", "num_partitioning_passes_needed",
        "sort_duration_breakdown", "total_sort_duration"
    ]
    output_mode = ["csv"]

    ####################################################################################################################
    # sorting_algorithm_to_sort_duration
    ####################################################################################################################

    identifier = "sorting_algorithm_to_sort_duration"
    arguments = []

    data_type = ["uint32"]
    distribution_type = ["uniform"]
    distribution_parameter = ["0"]

    algorithm = ["gnu-parallel-sort", "paradis-sort"]
    num_keys = [
        1000, 1000000, 500000000, 1000000000, 2000000000, 3000000000,
        4000000000, 6000000000, 8000000000, 12000000000, 16000000000,
        20000000000, 24000000000, 28000000000, 32000000000
    ]
    num_cpu_threads = ["128", "255"]

    arguments += list(
        itertools.product(*[
            algorithm, num_keys, num_cpu_threads, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    algorithm = ["thrust-sort"]
    num_keys = [
        1000, 1000000, 500000000, 1000000000, 2000000000, 3000000000,
        4000000000, 6000000000, 8000000000
    ]
    gpus = ["0"]

    arguments += list(
        itertools.product(*[
            algorithm, num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    algorithm = ["radix-mgpu-sort"]
    num_keys = [
        1000, 1000000, 500000000, 1000000000, 2000000000, 3000000000,
        4000000000, 6000000000, 8000000000
    ]
    gpus = ["0,1", "0,2,4,6", "0,1,2,3,4,5,6,7"]

    arguments += list(
        itertools.product(*[
            algorithm, num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    algorithm = ["radix-mgpu-sort"]
    num_keys = [
        12000000000, 16000000000, 20000000000, 24000000000, 28000000000,
        32000000000
    ]
    gpus = ["0,1,2,3,4,5,6,7"]

    arguments += list(
        itertools.product(*[
            algorithm, num_keys, gpus, data_type, distribution_type,
            distribution_parameter, output_mode
        ]))

    experiments.append(Experiment(identifier, executable, arguments, columns))
