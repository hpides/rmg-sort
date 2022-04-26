import pathlib
import subprocess
import time
import sys

import benchmark

if __name__ == "__main__":

    build_folder = "build"

    if len(sys.argv) >= 2:
        build_folder = sys.argv[1]

    benchmark.InitExperiments()

    executables_path = "../" + build_folder

    script_path = pathlib.Path(__file__).parent.resolve()
    executable_path = pathlib.Path(script_path / executables_path).resolve()
    output_path = pathlib.Path(script_path / benchmark.experiments_path /
                               time.strftime("%Y_%m_%d_%H_%M_%S")).resolve()
    profiler_output_path = pathlib.Path(output_path / "profiler").resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    profiler_output_path.mkdir(parents=True, exist_ok=True)

    experiments_to_run = [
        # "num_keys_to_sort_duration", "data_type_to_sort_duration",
        # "distribution_type_to_sort_duration",
        # "num_keys_to_sort_duration_profile_nsys",
        "sorting_algorithm_to_sort_duration",
        # "distribution_type_to_sort_duration_profile_nsys"
    ]

    for experiment in benchmark.experiments:

        if experiments_to_run and experiment.identifier not in experiments_to_run:
            continue

        for index, arguments in enumerate(experiment.arguments):
            numactl = "numactl -m 0" if (
                "gnu-parallel-sort" in arguments
                or "paradis-sort" in arguments) else "numactl -N 0 -m 0"

            command = "%s %s" % (numactl,
                                 pathlib.Path(executable_path /
                                              experiment.executable).resolve())

            for argument in arguments:
                command += " %s " % argument

            for repetition in range(experiment.repetitions):
                columns = "%s\n" % (",".join("\"%s\"" % (column)
                                             for column in experiment.columns)
                                    ) if index == 0 and repetition == 0 else ""

                if experiment.profilers:
                    for profiler in experiment.profilers:
                        if profiler == "nvprof" or profiler == "nsys":
                            output_file = pathlib.Path(
                                profiler_output_path /
                                ("%s_%s_%s_%s_%s_%s_%s_%s" %
                                 (experiment.executable, experiment.identifier,
                                  arguments[0], arguments[1], arguments[2],
                                  arguments[3], arguments[4], repetition + 1))
                            ).resolve()

                            if profiler == "nvprof":
                                command = ";".join([
                                    "nvprof --csv --log-file %s.csv %s" %
                                    (output_file, command),
                                    "nvprof --quiet --export-profile %s.nvvp %s"
                                    % (output_file, command)
                                ])
                            elif profiler == "nsys":
                                command = "nsys profile -o %s.qdrep %s" % (
                                    output_file, command)

                            output = subprocess.run(command,
                                                    stdout=subprocess.PIPE,
                                                    universal_newlines=True,
                                                    shell=True)

                            if output.returncode == 0:
                                print("%s%s" % (columns, output.stdout),
                                      end="")

                else:
                    output = subprocess.run(command,
                                            stdout=subprocess.PIPE,
                                            universal_newlines=True,
                                            shell=True)

                    if output.returncode == 0:
                        print("%s%s" % (columns, output.stdout), end="")

                        output_file = pathlib.Path(
                            output_path /
                            ("%s_%s.csv" %
                             (experiment.executable, experiment.identifier))
                        ).resolve()

                        with output_file.open("a") as f:
                            f.write("%s%s" % (columns, output.stdout))
