import subprocess
import sys
import os
import itertools

if __name__ == "__main__":

    executable = "radix-mgpu-sort"
    max_num_gpus = 4  # Set the maximum number of GPUs available on the system
    build_folder = "build"  # Set the name of the build folder, i.e. where the executable lies
    test_type = "default"  # Set the test type from {default, skew, custom, other}
    numa_node = 0  # Set the NUMA node from where to run

    if len(sys.argv) >= 2:
        max_num_gpus = int(sys.argv[1])

    if len(sys.argv) >= 3:
        build_folder = sys.argv[2]

    if len(sys.argv) >= 4:
        test_type = sys.argv[3]

    if len(sys.argv) >= 5:
        numa_node = sys.argv[4]

    os.chdir(os.getcwd() + "/" + build_folder)

    arguments = []

    if (test_type == "default"):
        num_keys = []
        for i in range(1, 1000):
            num_keys.append(i)

        for i in range(1, 25):
            for j in range(-3, 3):
                num_keys.append((1000 * i) + j)

        gpus = ["0,1", "0,1,2", "0,2,3,1", "0,1,2,3,4,5", "0,1,2,3,4,5,6,7"]
        data_type = ["uint32", "uint64", "float32", "float64"]
        data_distribution = ["uniform"]

        for g in list(gpus):
            if (g.count(',') >= max_num_gpus):
                gpus.remove(g)

        arguments += list(
            itertools.product(*[num_keys, gpus, data_type, data_distribution]))
    elif (test_type == "skew"):
        num_keys = []
        for i in range(1, 500):
            num_keys.append(i)

        for i in range(1, 50):
            num_keys.append(20000 + i)

        gpus = ["0,1", "0,1,3", "0,2,3,1", "0,2,4,6,7,5,3,1"]
        data_type = ["uint32"]

        bit_entropy = [
            "32", "28", "24", "16", "15", "9", "8", "7", "5", "2", "1"
        ]
        zipf_exponent = ["1.0", "0.75", "0.5", "0.25", "0.1", "0.0"]

        for g in list(gpus):
            if (g.count(',') >= max_num_gpus):
                gpus.remove(g)

        arguments += list(
            itertools.product(
                *[num_keys, gpus, data_type, ["skewed"], bit_entropy]))
        arguments += list(
            itertools.product(
                *[num_keys, gpus, data_type, ["zipf"], zipf_exponent]))
    elif (test_type == "custom"):
        num_keys = ["20"]

        gpus = ["0,1", "0,1,2", "0,1,2,3", "0,1,2,3,4,5,6,7"]

        for g in list(gpus):
            if (g.count(',') >= max_num_gpus):
                gpus.remove(g)

        data_distribution = ["custom"]

        data_type = ["uint32", "uint64"]
        file_name = [
            "../data/epsilon.csv", "../data/lastpass.csv",
            "../data/reverse.csv"
        ]

        arguments += list(
            itertools.product(
                *[num_keys, gpus, data_type, data_distribution, file_name]))

        data_type = ["float32", "float64"]
        file_name = ["../data/floats.csv"]

        arguments += list(
            itertools.product(
                *[num_keys, gpus, data_type, data_distribution, file_name]))
    elif (test_type == "other"):
        num_keys = []
        for i in range(1, 1000):
            num_keys.append(i)

        for i in range(1, 10):
            for j in range(-1, 2):
                num_keys.append((1000 * i) + j)

        gpus = ["0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]
        data_type = ["uint32"]
        data_distribution = [
            "sorted", "nearly-sorted", "reverse-sorted", "normal", "zero"
        ]

        for g in list(gpus):
            if (g.count(',') >= max_num_gpus):
                gpus.remove(g)

        arguments += list(
            itertools.product(*[num_keys, gpus, data_type, data_distribution]))

    commands = []

    for argument in arguments:
        command = "numactl -N " + str(numa_node) + " -m " + str(
            numa_node) + " ./" + executable
        for i in range(0, len(argument)):
            command += " "
            command += str(argument[i])
        commands.append(command)

    tests_failed = 0
    tests_succeeded = 0
    test_runs = 0
    print("Running {} tests:".format(len(commands)))

    for command in commands:
        test_runs += 1
        output = subprocess.run(command,
                                stdout=subprocess.PIPE,
                                universal_newlines=True,
                                shell=True)
        if output.returncode != 0:
            print("\x1B[31m%s\033[0m\n" % (command), end="")
            tests_failed += 1
        else:
            tests_succeeded += 1

        if tests_succeeded % 50 == 0:
            print("\x1B[32m%s\033[0m tests succeeded and counting...\n" %
                  (tests_succeeded),
                  end="")

        if tests_failed % 10 == 0 and tests_failed > 0 and output.returncode != 0:
            print("\x1B[31m%s\033[0m tests failed already...\n" %
                  (tests_failed),
                  end="")

    print("Ran %s tests!\n" % (test_runs), end="")
    print("\x1B[31m%s\033[0m tests failed!\n" % (tests_failed), end="")
    print("\x1B[32m%s\033[0m tests succeeded!\n" % (tests_succeeded), end="")
