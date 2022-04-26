from cmath import exp
from time import time
import matplotlib as plotlib
import matplotlib.pyplot as pyplot
import matplotlib.container as pycontainer
import numpy
import pandas
import pathlib
import sys

import benchmark

from typing import Dict, List, Union

plotlib.rcParams['hatch.linewidth'] = 0.5
plotlib.rcParams["mathtext.fontset"] = "cm"
plotlib.rc("font", **{"family": "Linux Libertine O"})

figure_width = 4.85
figure_height = 3.85


def scale_figure_size(width_factor: float, height_factor: float):
    pyplot.figure(num=1,
                  figsize=(figure_width * width_factor,
                           figure_height * height_factor))


legend_font_size = 18
small_font_size = 20
large_font_size = 22

colors = ["#006ddb", "#009292", "#db6d00", "#920000", "#ff6db6"]
hatches = ["//", "\\\\", "xx", "...", "oo"]
markers = ["o", "v", "s", "d", "h"]


def calculate_means(rows: List[str]):
    return [
        numpy.array([[float(value) for value in row.split(",")]
                     for row in rows]).mean(axis=0)
    ]


def annotate_bars(bars: pycontainer.BarContainer,
                  precision: int,
                  height: float = None,
                  rotation: str = None):
    bar = bars[0]
    bar_height = height or bar.get_height()

    value = str(int(bar_height)) if precision == 0 else (
        "{:.%sf}" % (precision)).format(bar_height)

    pyplot.annotate(value,
                    xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                    xytext=(0, 5 if rotation is not None else 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=small_font_size,
                    zorder=3)


def plot_bars(data: Dict[str, List[Union[float, int]]],
              precision: int,
              colors: List[str],
              hatches: List[str] = None,
              single_width: float = 0.9,
              total_width: float = 0.8,
              label_rotation: str = None):
    bars = []

    num_bars = len(data)
    bar_width = total_width / num_bars

    for index, values in enumerate(data.values()):
        x_offset = (index - num_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = pyplot.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[index % len(colors)],
                hatch=hatches[index %
                              len(hatches)] if hatches is not None else None,
                alpha=0.99 if hatches is not None else 1,
                zorder=2)

            if x == 0:
                bars.append(bar[0])

            annotate_bars(bar, precision, rotation=label_rotation)

    labels = data.keys()

    pyplot.legend(bars,
                  labels,
                  fontsize=legend_font_size,
                  loc='center left',
                  bbox_to_anchor=(1.0, 0.5))


def plot_lines(x_values, y_values, colors, markers, labels):
    zorder = 2 + len(x_values)
    for index in range(len(x_values)):
        pyplot.plot(x_values[index],
                    y_values[index],
                    linestyle="-",
                    color=colors[index],
                    marker=markers[index],
                    markersize=4,
                    label=labels[index],
                    zorder=zorder)

        zorder -= 1


def plot_stacked_bars(segment_columns, segment_colors, segment_hatches,
                      segment_labels, column, precision):
    handles = []

    for column_index, column_value in enumerate(data[column].tolist()):
        column_data = data[data[column] == column_value]

        bottom = 0
        for segment_index, segment_column in enumerate(segment_columns):
            bar = pyplot.bar(
                column_value,
                column_data[segment_columns[segment_index]].tolist()[0],
                bottom=bottom,
                color=segment_colors[segment_index % len(segment_colors)],
                hatch=segment_hatches[segment_index % len(segment_hatches)]
                if segment_hatches is not None else None,
                alpha=0.99 if segment_hatches is not None else 1,
                label=segment_labels[segment_index]
                if column_index == 0 else "",
                zorder=2)

            if column_index == 0:
                handles.append(bar)

            bottom += column_data[segment_columns[segment_index]].tolist()[0]

            if segment_index == len(segment_columns) - 1:
                annotate_bars(bar, precision, bottom)

    return handles


def configure_plot(x_ticks_ticks=None,
                   x_ticks_labels=None,
                   y_ticks_ticks=None,
                   y_ticks_labels=None,
                   x_ticks_rotation=False,
                   x_ticks_minor=False,
                   x_label=None,
                   y_label=None,
                   legend=False,
                   legend_handles=None,
                   legend_columns=None):

    pyplot.gca().yaxis.offsetText.set_fontsize(legend_font_size)
    pyplot.gca().xaxis.offsetText.set_fontsize(legend_font_size)

    pyplot.xticks(fontsize=small_font_size)
    if x_ticks_ticks is not None and x_ticks_labels is not None:
        pyplot.xticks(ticks=x_ticks_ticks, labels=x_ticks_labels)
    elif x_ticks_ticks is not None:
        pyplot.xticks(ticks=x_ticks_ticks)

    pyplot.yticks(fontsize=small_font_size)
    if y_ticks_ticks is not None and y_ticks_labels is not None:
        pyplot.yticks(ticks=y_ticks_ticks, labels=y_ticks_labels)
    elif y_ticks_ticks is not None:
        pyplot.yticks(ticks=y_ticks_ticks)

    if x_ticks_rotation:
        pyplot.xticks(rotation=45, ha="right")

    pyplot.minorticks_on()
    if not x_ticks_minor:
        pyplot.tick_params(axis='x', which='minor', bottom=False)

    if x_label is not None:
        pyplot.xlabel(x_label, fontsize=large_font_size)

    if y_label is not None:
        pyplot.ylabel(y_label, fontsize=large_font_size)

    if legend and legend_handles is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      loc='center left',
                      handles=legend_handles,
                      ncol=legend_columns,
                      labelspacing=0.4,
                      bbox_to_anchor=(1.0, 0.5))
    elif legend and legend_handles is not None:
        pyplot.legend(fontsize=legend_font_size,
                      loc='center left',
                      handles=legend_handles,
                      labelspacing=0.4,
                      bbox_to_anchor=(1.0, 0.5))
    elif legend:
        pyplot.legend(fontsize=legend_font_size,
                      loc='center left',
                      labelspacing=0.4,
                      bbox_to_anchor=(1.0, 0.5))


def parseSortDurationBreakdown(sort_duration_breakdown_str,
                               num_partition_passes_needed, num_gpus):

    # Returns a list of time durations, one for each algorithm phase.
    # The time duration breakdown consists of the following algporithm phases:
    # HtoD, Radix-Partition (ComputeHistogram + ScatterKeys), P2PKeySwap, SortChunk, SortBuckets-and-CopyBack, DtoH
    # Each GPU executes these phases concurrently. For each algorithm phase, the one that took the longest time is selected.
    # Multiple partitioning passes result in multiple runs of the ComputeHistogram and ScatterKeys kernels whose time durations are added up.

    sort_duration_breakdown_str = sort_duration_breakdown_str.replace("{", "")
    sort_duration_breakdown_str = sort_duration_breakdown_str.replace("}", "")
    sort_duration_breakdown_str = sort_duration_breakdown_str.replace("(", "")
    sort_duration_breakdown_str = sort_duration_breakdown_str.replace(")", "")

    if num_gpus == 1:
        time_durations = [
            float(time_duration_str)
            for time_duration_str in sort_duration_breakdown_str.split(",")
        ]
        return [
            time_durations[0], 0, time_durations[1], time_durations[2], 0,
            time_durations[3]
        ]
    else:
        max_durations_per_gpu = []
        num_fields_per_gpu = 4 + 2 * num_partition_passes_needed
        all_time_durations = [
            float(time_duration_str)
            for time_duration_str in sort_duration_breakdown_str.split(",")
        ]

        for g in range(num_gpus):
            for i in range(num_fields_per_gpu):

                time = all_time_durations[num_fields_per_gpu * g + i]
                if len(max_durations_per_gpu) < i + 1:
                    max_durations_per_gpu.append(time)
                elif time > max_durations_per_gpu[i]:
                    max_durations_per_gpu[i] = time

        total_histo_time = 0
        total_scatter_time = 0
        for i in range(num_partition_passes_needed):
            total_histo_time += max_durations_per_gpu[i + 1]
            total_scatter_time += max_durations_per_gpu[
                i + num_partition_passes_needed + 1]

        jump_offset = 2 * num_partition_passes_needed
        return [
            max_durations_per_gpu[0], total_histo_time + total_scatter_time,
            max_durations_per_gpu[1 + jump_offset], 0,
            max_durations_per_gpu[2 + jump_offset],
            max_durations_per_gpu[3 + jump_offset]
        ]


if __name__ == "__main__":
    benchmark.InitExperiments()

    benchmark_results_folder = ""
    if (len(sys.argv) != 2):
        print(
            "Error: Specify the name of the folder that contains the benchmark results to plot."
        )
        exit(1)
    else:
        benchmark_results_folder = sys.argv[1]

    experiments_to_plot = [
        ("num_keys_to_sort_duration",
         "num_keys_to_sort_duration_bar_per_gpus"),
        ("num_keys_to_sort_duration",
         "num_keys_to_sort_duration_for_gpus_line_plot"),
        ("num_keys_to_sort_duration", "gpus_to_sort_duration_breakdown"),
        ("distribution_type_to_sort_duration",
         "distribution_type_to_sort_duration_overview"),
        ("distribution_type_to_sort_duration",
         "skewed_bit_entropy_distribution_to_sort_duration"),
        ("distribution_type_to_sort_duration",
         "zipf_exponent_distribution_to_sort_duration"),
        ("data_type_to_sort_duration", "data_type_to_sort_duration_overview"),
        ("sorting_algorithm_to_sort_duration",
         "sorting_algorithm_to_sort_duration_bar_per_algorithm"),
        ("sorting_algorithm_to_sort_duration",
         "num_keys_to_sort_duration_vs_cpu_line_plot")
    ]

    script_path = pathlib.Path(__file__).parent.resolve()
    benchmark_results_path = pathlib.Path(script_path /
                                          benchmark.experiments_path /
                                          benchmark_results_folder).resolve()

    for (experiment_id, plot_id) in experiments_to_plot:
        print("Plot " + plot_id)

        try:
            experiment = next(experiment
                              for experiment in benchmark.experiments
                              if experiment.identifier == experiment_id)
        except StopIteration:
            print("Error: Unable plot " + plot_id + ". The experiment " +
                  experiment_id + " is not defined in the benchmark.")
            continue

        ############################################################################################################
        # num_keys_to_sort_duration_bar_per_gpus
        ############################################################################################################

        if plot_id == "num_keys_to_sort_duration_bar_per_gpus":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby(["num_keys", "gpus"], as_index=False).mean()

            max_gpu_set = max(data["gpus"].tolist(), key=len)
            max_num_gpus = max_gpu_set.count(",") + 1
            data = data[(data["num_keys"] == 2000000000)]

            gpu_sets = data["gpus"].tolist()

            best_gpu_set_per_num_gpus = {}
            for g in range(1, max_num_gpus + 1):
                best_gpu_set_per_num_gpus[g] = None

            for gpu_set in gpu_sets:
                total_sort_duration = data[(
                    data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                num_gpus = gpu_set.count(",") + 1

                if best_gpu_set_per_num_gpus[num_gpus] == None:
                    best_gpu_set_per_num_gpus[num_gpus] = total_sort_duration
                elif total_sort_duration < best_gpu_set_per_num_gpus[num_gpus]:
                    best_gpu_set_per_num_gpus[num_gpus] = total_sort_duration

            gpu_set_to_sort_duration = {}
            bar_colors = []
            bar_hatches = []

            max_plot_height = 0
            color_count = 0
            for g in best_gpu_set_per_num_gpus:
                if best_gpu_set_per_num_gpus[g] != None:
                    label = str(g) + " GPU(s)"
                    gpu_set_to_sort_duration[label] = [
                        best_gpu_set_per_num_gpus[g]
                    ]
                    if best_gpu_set_per_num_gpus[g] > max_plot_height:
                        max_plot_height = best_gpu_set_per_num_gpus[g]
                    bar_colors.append(colors[color_count])
                    bar_hatches.append(hatches[color_count])
                    color_count += 1

            scale_figure_size(1.75, 1)

            plot_bars(gpu_set_to_sort_duration, 1, bar_colors, bar_hatches)

            ticks = ["2"]
            max_plot_height += 0.25 * max_plot_height
            max_plot_height = round(max_plot_height / 100) * 100

            y_ticks_ticks = numpy.arange(0, max_plot_height + 100, 100)

            configure_plot(x_ticks_ticks=range(len(ticks)),
                           x_ticks_labels=ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="Number of keys [1e9]",
                           y_label="Sort duration [ms]")

        ############################################################################################################
        # num_keys_to_sort_duration_for_gpus_line_plot
        ############################################################################################################

        elif plot_id == "num_keys_to_sort_duration_for_gpus_line_plot":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby(["num_keys", "gpus"], as_index=False).mean()

            gpu_sets = data["gpus"].tolist()
            _data = data[(data["num_keys"] == 2000000000)]

            best_gpu_set_per_num_gpus = {}
            for g in range(1, max_num_gpus + 1):
                best_gpu_set_per_num_gpus[g] = None

            for gpu_set in gpu_sets:
                total_sort_duration = _data[(
                    _data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                num_gpus = gpu_set.count(",") + 1
                data_points = [
                    data_point / 1000.0
                    for data_point in data[data["gpus"] == gpu_set]
                    ["total_sort_duration"].tolist()
                ]

                if best_gpu_set_per_num_gpus[num_gpus] == None:
                    best_gpu_set_per_num_gpus[num_gpus] = (total_sort_duration,
                                                           data_points,
                                                           gpu_set)
                elif total_sort_duration < best_gpu_set_per_num_gpus[num_gpus][
                        0]:
                    best_gpu_set_per_num_gpus[num_gpus] = (total_sort_duration,
                                                           data_points,
                                                           gpu_set)

            line_markers = []
            line_labels = []
            line_colors = []
            x_values = []
            y_values = []

            data["num_keys"] = [
                num_keys / 1000000000 for num_keys in data["num_keys"]
            ]

            color_count = 0
            for g in best_gpu_set_per_num_gpus:
                if best_gpu_set_per_num_gpus[g] != None:
                    line_markers.append(markers[color_count % len(markers)])
                    line_labels.append(str(g) + " GPU(s)")
                    line_colors.append(colors[color_count % len(colors)])

                    data_points = best_gpu_set_per_num_gpus[g][1]
                    gpu_set = best_gpu_set_per_num_gpus[g][2]
                    x_values.append(
                        data[data["gpus"] == gpu_set]["num_keys"].tolist())
                    y_values.append(data_points)
                    color_count += 1

            scale_figure_size(1.5, 1)

            plot_lines(x_values, y_values, line_colors, line_markers,
                       line_labels)

            configure_plot(y_ticks_ticks=numpy.arange(0, 3.6, 0.4),
                           x_label="Number of keys [1e9]",
                           y_label="Sort duration [ms]",
                           legend=True)

        ############################################################################################################
        # sorting_algorithm_to_sort_duration_bar_per_algorithm
        ############################################################################################################

        elif plot_id == "sorting_algorithm_to_sort_duration_bar_per_algorithm":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby(
                ["sorting_algorithm", "num_keys", "gpus", "num_cpu_threads"],
                as_index=False).mean()

            data = data[(data["num_keys"] == 2000000000)]
            max_num_cpu_threads = max(data["num_cpu_threads"].tolist())
            cpu_data = data[(data["num_cpu_threads"] == max_num_cpu_threads)]

            gpu_sets = data["gpus"].tolist()

            max_gpu_set = max(gpu_sets, key=len)
            max_num_gpus = max_gpu_set.count(",") + 1

            best_sort_duration_map = {}

            data_ = cpu_data[(cpu_data["gpus"] == "-")]
            sorting_algorithms = data_["sorting_algorithm"].tolist()

            max_plot_height = 0
            # cpu
            for algorithm in sorting_algorithms:
                total_sort_duration = data_[
                    (data_["sorting_algorithm"] == algorithm
                     )]["total_sort_duration"].tolist()[0] / 1000.0

                if total_sort_duration > max_plot_height:
                    max_plot_height = total_sort_duration
                label = algorithm.replace("-sort", ", CPU")
                best_sort_duration_map[label] = total_sort_duration

            # thrust
            total_sort_duration = data[
                (data["sorting_algorithm"] == "thrust-sort"
                 )]["total_sort_duration"].tolist()[0] / 1000.0
            label = "thrust, 1 GPU(s)"
            best_sort_duration_map[label] = total_sort_duration

            # radix-mgpu
            for g in range(2, max_num_gpus + 1):
                best_sort_duration_map["radix-mgpu, " + str(g) +
                                       " GPU(s)"] = None

            for gpu_set in gpu_sets:
                total_sort_duration = data[
                    (data["gpus"] == gpu_set
                     )]["total_sort_duration"].tolist()[0] / 1000.0

                if total_sort_duration > max_plot_height:
                    max_plot_height = total_sort_duration

                num_gpus = 0
                num_gpus = gpu_set.count(",") + 1
                if num_gpus >= 2:

                    algorithm = data[(data["gpus"] == gpu_set
                                      )]["sorting_algorithm"].tolist()[0]
                    label = algorithm.replace("-sort", ", ")
                    label += str(num_gpus) + " GPU(s)"
                    if best_sort_duration_map[label] == None:
                        best_sort_duration_map[label] = total_sort_duration
                    elif total_sort_duration < best_sort_duration_map[label]:
                        best_sort_duration_map[label] = total_sort_duration

            gpu_set_to_sort_duration = {}
            bar_colors = []
            bar_hatches = []

            color_count = 0
            for label in best_sort_duration_map:
                if best_sort_duration_map[label] != None:
                    color = colors[color_count % len(colors)]
                    gpu_set_to_sort_duration[label] = [
                        best_sort_duration_map[label]
                    ]
                    bar_colors.append(color)
                    color_count += 1

            bar_hatches = [
                hatches[0], hatches[2], hatches[1], hatches[2], hatches[3],
                hatches[4]
            ]

            scale_figure_size(2, 1)

            plot_bars(gpu_set_to_sort_duration, 3, bar_colors, bar_hatches)

            ticks = ["2"]

            max_plot_height += 2.0
            max_plot_height = round(max_plot_height)

            if max_plot_height / 10 > 1.0:
                y_ticks_ticks = numpy.arange(0, max_plot_height, 2.0)
            else:
                y_ticks_ticks = numpy.arange(0, max_plot_height, 1.0)

            configure_plot(x_ticks_ticks=range(len(ticks)),
                           x_ticks_labels=ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="Number of keys [1e9]",
                           y_label="Sort duration [s]")

        ############################################################################################################
        # num_keys_to_sort_duration_vs_cpu_line_plot
        ############################################################################################################

        elif plot_id == "num_keys_to_sort_duration_vs_cpu_line_plot":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby(
                ["sorting_algorithm", "num_keys", "gpus", "num_cpu_threads"],
                as_index=False).mean()

            data["num_keys"] = [
                int(num_keys / 1000000000) for num_keys in data["num_keys"]
            ]

            max_num_cpu_threads = max(data["num_cpu_threads"].tolist())
            cpu_data = data[(data["num_cpu_threads"] == max_num_cpu_threads)]

            gpu_sets = data["gpus"].tolist()
            _data = data[(data["num_keys"] == 2)]
            _cpu_data = cpu_data[(cpu_data["num_keys"] == 2)]
            sorting_algorithms = _cpu_data[(
                _cpu_data["gpus"] == "-")]["sorting_algorithm"].tolist()

            cpu_data_points = []
            for algorithm in sorting_algorithms:

                data_points = [
                    data_point / 1000.0 for data_point in cpu_data[
                        cpu_data["sorting_algorithm"] == algorithm]
                    ["total_sort_duration"].tolist()
                ]
                cpu_data_points.append((data_points, algorithm))

            best_gpu_set = None

            for gpu_set in gpu_sets:
                total_sort_duration = _data[(
                    _data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                algorithm = data[(
                    data["gpus"] == gpu_set)]["sorting_algorithm"].tolist()[0]

                num_gpus = gpu_set.count(",") + 1
                if num_gpus >= 2:
                    data_points = [
                        data_point / 1000.0
                        for data_point in data[data["gpus"] == gpu_set]
                        ["total_sort_duration"].tolist()
                    ]

                if best_gpu_set == None:
                    best_gpu_set = (total_sort_duration, data_points, gpu_set,
                                    algorithm)
                elif total_sort_duration < best_gpu_set[0]:
                    best_gpu_set = (total_sort_duration, data_points, gpu_set,
                                    algorithm)

            line_markers = []
            line_labels = []
            line_colors = []
            x_values = []
            y_values = []

            max_plot_height = 0
            color_count = 0
            for data_points in cpu_data_points:
                algorithm = data_points[1]
                line_markers.append(markers[color_count % len(markers)])
                line_labels.append(algorithm.replace("-sort", ", CPU"))
                line_colors.append(colors[color_count % len(colors)])
                x_values.append(cpu_data[cpu_data["sorting_algorithm"] ==
                                         algorithm]["num_keys"].tolist())
                y_values.append(data_points[0])
                color_count += 1

            gpu_set = best_gpu_set[2]
            algorithm = best_gpu_set[3]
            num_gpus = gpu_set.count(",") + 1
            line_markers.append(markers[color_count % len(markers)])
            line_labels.append(
                algorithm.replace("-sort", ", " + str(num_gpus) + " GPU(s)"))
            line_colors.append(colors[color_count % len(colors)])
            algorithm_data = data[(data["sorting_algorithm"] == algorithm)]
            x_values.append(algorithm_data[(
                algorithm_data["gpus"] == gpu_set)]["num_keys"].tolist())
            y_values.append(best_gpu_set[1])

            max_x_value_points = 1000
            for x_value_list in x_values:
                if len(x_value_list) < max_x_value_points:
                    max_x_value_points = len(x_value_list)

            for y_value_list in y_values:
                del y_value_list[max_x_value_points:]

            for x_value_list in x_values:
                del x_value_list[max_x_value_points:]

            for y_value_list in y_values:
                if y_value_list[-1] > max_plot_height:
                    max_plot_height = y_value_list[-1]

            scale_figure_size(2, 1)

            plot_lines(x_values, y_values, line_colors, line_markers,
                       line_labels)

            max_plot_height += 0.1 * max_plot_height
            max_plot_height = round(max_plot_height / 10) * 10
            max_plot_height += 10

            scale_increase = 2
            if max_plot_height / scale_increase >= 20:
                scale_increase *= 5

            configure_plot(
                y_ticks_ticks=numpy.arange(
                    0, max_plot_height,
                    scale_increase),  #numpy.arange(0, 20, 4),
                x_label="Number of keys [1e9]",
                y_label="Sort duration [ms]",
                legend=True)

        ############################################################################################################
        # distribution_type_to_sort_duration_overview
        ############################################################################################################

        elif plot_id == "distribution_type_to_sort_duration_overview":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby([
                "num_keys", "gpus", "distribution_type",
                "distribution_parameter"
            ],
                                as_index=False).mean()

            data["num_keys"] = [
                int(num_keys / 1000000000) for num_keys in data["num_keys"]
            ]
            data = data[(data["num_keys"] == 2)]
            data = data[(data["distribution_parameter"] == 0)]
            uniform_data = data[(data["distribution_type"] == "uniform")]
            data = data.sort_values("total_sort_duration")

            gpu_sets = data["gpus"].tolist()
            best_gpu_set = None

            for gpu_set in gpu_sets:
                total_sort_duration = uniform_data[(
                    uniform_data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                if best_gpu_set == None:
                    best_gpu_set = (total_sort_duration, gpu_set)
                elif total_sort_duration < best_gpu_set[0]:
                    best_gpu_set = (total_sort_duration, gpu_set)

            data = data[(data["gpus"] == best_gpu_set[1])]
            distribution_types = data["distribution_type"].tolist()
            distribution_types.remove("zipf")
            distribution_types.remove("skewed")

            distribution_type_to_durations = {}
            for distribution_type in distribution_types:
                distribution_type_to_durations[distribution_type.capitalize(
                )] = data[data["distribution_type"] ==
                          distribution_type]["total_sort_duration"].tolist()

            scale_figure_size(2, 1)

            bar_hatches = [
                hatches[0], hatches[2], hatches[1], hatches[2], hatches[3],
                hatches[4]
            ]

            plot_bars(distribution_type_to_durations,
                      1,
                      colors,
                      bar_hatches,
                      label_rotation="horizontal")

            ticks = [""]

            y_ticks_ticks = y_ticks_ticks = numpy.arange(0, 300, 40)

            configure_plot(x_ticks_ticks=range(len(ticks)),
                           x_ticks_labels=ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="Distribution type",
                           y_label="Sort duration [ms]")

        ############################################################################################################
        # skewed_bit_entropy_distribution_to_sort_duration
        ############################################################################################################

        elif plot_id == "skewed_bit_entropy_distribution_to_sort_duration":
            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby([
                "num_keys", "gpus", "distribution_type",
                "distribution_parameter"
            ],
                                as_index=False).mean()

            data["num_keys"] = [
                int(num_keys / 1000000000) for num_keys in data["num_keys"]
            ]
            data = data[(data["num_keys"] == 2)]
            uniform_data = data[(data["distribution_type"] == "uniform")]
            skewed_data = data[(data["distribution_type"] == "skewed")]

            gpu_sets = data["gpus"].tolist()
            best_gpu_set = None

            for gpu_set in gpu_sets:
                total_sort_duration = uniform_data[(
                    uniform_data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                if best_gpu_set == None:
                    best_gpu_set = (total_sort_duration, gpu_set)
                elif total_sort_duration < best_gpu_set[0]:
                    best_gpu_set = (total_sort_duration, gpu_set)

            skewed_data_for_best_gpu = skewed_data[(
                skewed_data["gpus"] == best_gpu_set[1])]
            bit_entropies = skewed_data_for_best_gpu[
                "distribution_parameter"].tolist()

            max_plot_height = 0
            min_plot_height = 1000000000

            sort_durations = []
            for bit_entropy in bit_entropies:
                total_sort_duration = skewed_data_for_best_gpu[(
                    skewed_data_for_best_gpu["distribution_parameter"] ==
                    bit_entropy)]["total_sort_duration"].tolist()[0]

                if total_sort_duration < min_plot_height:
                    min_plot_height = total_sort_duration

                if total_sort_duration > max_plot_height:
                    max_plot_height = total_sort_duration

                sort_durations.append(total_sort_duration)

            num_gpus = best_gpu_set[1].count(",") + 1
            line_markers = [markers[0]]
            line_colors = [colors[0]]
            line_labels = [str(num_gpus) + "GPU(s)"]
            x_values = [bit_entropies]
            y_values = [sort_durations]

            scale_figure_size(2, 1)

            plot_lines(x_values, y_values, line_colors, line_markers,
                       line_labels)

            max_plot_height += 0.1 * max_plot_height
            max_plot_height = round(max_plot_height / 10) * 10
            min_plot_height -= 0.1 * min_plot_height
            min_plot_height = round(min_plot_height / 10) * 10

            scale_increase = 10
            if (max_plot_height - min_plot_height) / scale_increase >= 10.0:
                scale_increase *= 2

            configure_plot(y_ticks_ticks=numpy.arange(min_plot_height,
                                                      max_plot_height,
                                                      scale_increase),
                           x_ticks_ticks=numpy.arange(0, 33, 2),
                           x_label="Bit entropy",
                           y_label="Sort duration [ms]",
                           legend=False)

        ############################################################################################################
        # zipf_exponent_distribution_to_sort_duration
        ############################################################################################################

        elif plot_id == "zipf_exponent_distribution_to_sort_duration":
            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby([
                "num_keys", "gpus", "distribution_type",
                "distribution_parameter"
            ],
                                as_index=False).mean()

            data["num_keys"] = [
                int(num_keys / 1000000000) for num_keys in data["num_keys"]
            ]
            data = data[(data["num_keys"] == 2)]
            uniform_data = data[(data["distribution_type"] == "uniform")]
            zipf_data = data[(data["distribution_type"] == "zipf")]

            gpu_sets = data["gpus"].tolist()
            best_gpu_set = None
            max_plot_height = 0
            min_plot_height = 1000000000

            for gpu_set in gpu_sets:
                total_sort_duration = uniform_data[(
                    uniform_data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                if best_gpu_set == None:
                    best_gpu_set = (total_sort_duration, gpu_set)
                elif total_sort_duration < best_gpu_set[0]:
                    best_gpu_set = (total_sort_duration, gpu_set)

            skewed_data_for_best_gpu = zipf_data[(
                zipf_data["gpus"] == best_gpu_set[1])]
            zipf_exponents = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

            sort_durations = []
            for zipf_exponent in zipf_exponents:
                total_sort_duration = skewed_data_for_best_gpu[(
                    skewed_data_for_best_gpu["distribution_parameter"] ==
                    zipf_exponent)]["total_sort_duration"].tolist()[0]

                if total_sort_duration < min_plot_height:
                    min_plot_height = total_sort_duration

                if total_sort_duration > max_plot_height:
                    max_plot_height = total_sort_duration

                sort_durations.append(total_sort_duration)

            num_gpus = best_gpu_set[1].count(",") + 1
            line_markers = [markers[0]]
            line_colors = [colors[0]]
            line_labels = [str(num_gpus) + "GPU(s)"]
            x_values = [zipf_exponents]
            y_values = [sort_durations]

            scale_figure_size(1.5, 1)

            plot_lines(x_values, y_values, line_colors, line_markers,
                       line_labels)

            max_plot_height += 0.1 * max_plot_height
            max_plot_height = round(max_plot_height / 10) * 10
            min_plot_height -= 0.1 * min_plot_height
            min_plot_height = round(min_plot_height / 10) * 10

            scale_increase = 10
            if (max_plot_height - min_plot_height) / scale_increase >= 10:
                scale_increase *= 2

            configure_plot(y_ticks_ticks=numpy.arange(min_plot_height,
                                                      max_plot_height,
                                                      scale_increase),
                           x_ticks_ticks=zipf_exponents,
                           x_label="Zipf exponent",
                           y_label="Sort duration [ms]",
                           legend=False)

        ############################################################################################################
        # data_type_to_sort_duration_overview
        ############################################################################################################

        elif plot_id == "data_type_to_sort_duration_overview":
            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            data = data.groupby([
                "num_keys", "gpus", "distribution_type", "data_type",
                "distribution_parameter"
            ],
                                as_index=False).mean()

            data["num_keys"] = [
                int(num_keys / 1000000000) for num_keys in data["num_keys"]
            ]
            data = data[(data["num_keys"] == 2)]
            uniform_data = data[(data["distribution_type"] == "uniform")]
            data = data.sort_values("total_sort_duration")

            gpu_sets = data["gpus"].tolist()
            best_gpu_set = None

            for gpu_set in gpu_sets:
                total_sort_duration = uniform_data[(
                    uniform_data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                if best_gpu_set == None:
                    best_gpu_set = (total_sort_duration, gpu_set)
                elif total_sort_duration < best_gpu_set[0]:
                    best_gpu_set = (total_sort_duration, gpu_set)

            data = data[(data["gpus"] == best_gpu_set[1])]
            data_types = data["data_type"].tolist()
            distribution_types = data["distribution_type"].tolist()

            bar_hatches = []
            bar_colors = []

            type_to_durations = {}
            for data_type in ["uint32", "float32", "uint64", "float64"]:
                distribution_type = ""
                if "uint" in data_type:
                    distribution_type = "uniform"

                if "float" in data_type:
                    distribution_type = "zipf"

                _data = data[(data["distribution_type"] == distribution_type)]
                if "uint" in data_type:
                    key = "integer"
                    bar_hatches.append(hatches[0])
                    bar_colors.append(colors[1])
                elif "float" in data_type:
                    key = "floating point"
                    bar_hatches.append(hatches[1])
                    bar_colors.append(colors[2])

                if key not in type_to_durations.keys():
                    type_to_durations[key] = [0, 0]
                if "32" in data_type:
                    type_to_durations[key][0] = round(
                        _data[_data["data_type"] ==
                              data_type]["total_sort_duration"].tolist()[0])
                elif "64" in data_type:
                    type_to_durations[key][1] = round(
                        _data[_data["data_type"] ==
                              data_type]["total_sort_duration"].tolist()[0])

            scale_figure_size(2, 1)

            plot_bars(type_to_durations,
                      0,
                      bar_colors,
                      bar_hatches,
                      label_rotation="horizontal")

            ticks = ["32-bit", "64-bit"]

            y_ticks_ticks = y_ticks_ticks = numpy.arange(0, 700, 100)

            configure_plot(x_ticks_ticks=range(len(ticks)),
                           x_ticks_labels=ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="Data type",
                           y_label="Sort duration [ms]")

        ############################################################################################################
        # gpus_to_sort_duration_breakdown
        ############################################################################################################

        elif plot_id == "gpus_to_sort_duration_breakdown":

            try:
                data = pandas.read_csv(pathlib.Path(benchmark_results_path / (
                    "%s_%s.csv" %
                    (experiment.executable, experiment.identifier))).resolve(),
                                       header=0)
            except FileNotFoundError:
                continue

            avg_data = data.groupby(["num_keys", "gpus"],
                                    as_index=False).mean()

            max_gpu_set = max(avg_data["gpus"].tolist(), key=len)
            max_num_gpus = max_gpu_set.count(",") + 1
            avg_data = avg_data[(avg_data["num_keys"] == 2000000000)]
            data = data[(data["num_keys"] == 2000000000)]

            gpu_sets = avg_data["gpus"].tolist()

            best_gpu_set_per_num_gpus = {}
            for g in range(1, max_num_gpus + 1):
                best_gpu_set_per_num_gpus[g] = None

            for gpu_set in gpu_sets:
                total_sort_duration = avg_data[(
                    avg_data["gpus"] == gpu_set
                )]["total_sort_duration"].tolist()[0]

                num_gpus = gpu_set.count(",") + 1

                if best_gpu_set_per_num_gpus[num_gpus] == None:
                    best_gpu_set_per_num_gpus[num_gpus] = (total_sort_duration,
                                                           gpu_set)
                elif total_sort_duration < best_gpu_set_per_num_gpus[num_gpus][
                        0]:
                    best_gpu_set_per_num_gpus[num_gpus] = (total_sort_duration,
                                                           gpu_set)

            gpu_set_to_sort_duration = {}
            bar_colors = []
            bar_hatches = []

            scale_figure_size(2, 1)
            handles = []
            ticks = []

            segment_labels = [
                "HtoD Copy", "Radix Partition", "P2P Key Swap", "Sort Chunk",
                "Sort Buckets & DtoH Copy", "DtoH Copy"
            ]
            segment_colors = [
                colors[0], colors[2], colors[3], colors[4], colors[1],
                colors[1]
            ]
            segment_hatches = [
                hatches[0], hatches[2], hatches[4], hatches[3], hatches[2],
                hatches[1]
            ]

            max_plot_height = 0
            for g in best_gpu_set_per_num_gpus:
                if best_gpu_set_per_num_gpus[g] != None:
                    ticks.append(g)
                    label = str(g) + " GPU(s)"
                    gpu_set = best_gpu_set_per_num_gpus[g][1]
                    sort_duration_breakdown_str = data[(
                        data["gpus"] == gpu_set
                    )]["sort_duration_breakdown"].tolist()[0]
                    num_partition_passes_needed = data[(
                        data["gpus"] == gpu_set
                    )]["num_partitioning_passes_needed"].tolist()[0]

                    time_durations = parseSortDurationBreakdown(
                        sort_duration_breakdown_str,
                        num_partition_passes_needed, g)

                    bottom = 0
                    for index, time_duration in enumerate(time_durations):
                        bar = pyplot.bar(label,
                                         time_duration,
                                         bottom=bottom,
                                         color=segment_colors[index],
                                         hatch=segment_hatches[index],
                                         alpha=1,
                                         label=segment_labels[index],
                                         zorder=2)

                        handles.append(bar)

                        bottom += time_duration

                        if index == len(time_durations) - 1:
                            annotate_bars(bar, 1, bottom)
                            if bottom > max_plot_height:
                                max_plot_height = bottom

            max_plot_height += 0.25 * max_plot_height
            max_plot_height = round(max_plot_height / 100) * 100

            y_ticks_ticks = numpy.arange(0, max_plot_height + 100, 100)

            configure_plot(x_ticks_ticks=range(len(ticks)),
                           x_ticks_labels=ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="Number of GPUs",
                           y_label="Sort duration [ms]",
                           legend=True,
                           legend_handles=handles[0:len(segment_colors)],
                           legend_columns=1)

        else:
            continue

        pyplot.tight_layout()
        pyplot.savefig(pathlib.Path(benchmark_results_path /
                                    ("%s.pdf" % (plot_id))).resolve(),
                       format="pdf")
        pyplot.close()
