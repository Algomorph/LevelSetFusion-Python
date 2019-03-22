#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 12/3/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
# stdlib
import sys
import os.path
import shutil
import re

# libraries
import numpy as np
import progressbar
import pandas as pd

# local
import utils.path as pu
import experiment.experiment_shared_routines as esr
from tsdf import generation as tsdf
from nonrigid_opt import field_warping as resampling
import experiment.hierarchical_optimizer.build_helper as build_opt
import utils.visualization as viz
import nonrigid_opt.hierarchical.hierarchical_optimization_visualizer as ho_viz
from experiment.hierarchical_optimizer.multipair_arguments import Arguments, post_process_enum_args
import nonrigid_opt.slavcheva.sobolev_filter as sob
from ext_argparse.argproc import process_arguments

# has to be compiled and installed first (cpp folder)
import level_set_fusion_optimization as cpp_module

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        else:
            shutil.rmtree(file_path)


def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            clear_folder(folder_path)
        else:
            raise ValueError("Path " + folder_path + " is not a folder.")
    else:
        os.makedirs(folder_path)


def create_folder_if_necessary(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    elif not os.path.isdir(folder_path):
        raise ValueError("Path " + folder_path + " is not a folder.")


digit_regex = re.compile(r'\d+')


def infer_frame_number_from_filename(filename):
    return int(digit_regex.findall(filename)[0])


def infer_frame_number_and_pixel_row_from_filename(filename):
    match_result = digit_regex.findall(filename)
    return int(match_result[0]), int(match_result[1])


def post_process_convergence_report_sets(convergence_report_sets, frame_numbers_and_rows):
    data = {}
    for i_level in range(len(convergence_report_sets[0])):
        data["canonical_frame"] = []
        data["pixel_row"] = []
        data["l" + str(i_level) + "_iter_count"] = []
        data["l" + str(i_level) + "_iter_lim_reached"] = []
        data["l" + str(i_level) + "_warp_delta_amt_ratio"] = []
        data["l" + str(i_level) + "_warp_delta_min"] = []
        data["l" + str(i_level) + "_warp_delta_max"] = []
        data["l" + str(i_level) + "_warp_delta_mean"] = []
        data["l" + str(i_level) + "_warp_delta_std"] = []
        data["l" + str(i_level) + "_warp_delta_max_x"] = []
        data["l" + str(i_level) + "_warp_delta_max_y"] = []
        data["l" + str(i_level) + "_warps_below_min_thresh"] = []
        data["l" + str(i_level) + "_warps_above_max_thresh"] = []
        data["l" + str(i_level) + "_diff_delta_min"] = []
        data["l" + str(i_level) + "_diff_delta_max"] = []
        data["l" + str(i_level) + "_diff_delta_mean"] = []
        data["l" + str(i_level) + "_diff_delta_std"] = []
        data["l" + str(i_level) + "_diff_max_x"] = []
        data["l" + str(i_level) + "_diff_max_y"] = []

    for report_set, (frame_number, pixel_row) in zip(convergence_report_sets, frame_numbers_and_rows):
        data["canonical_frame"].append(frame_number)
        data["pixel_row"].append(pixel_row)
        for i_level, report in enumerate(report_set):
            wds = report.warp_delta_statistics
            tds = report.tsdf_difference_statistics
            data["l" + str(i_level) + "_iter_count"].append(report.iteration_count)
            data["l" + str(i_level) + "_iter_lim_reached"].append(report.iteration_limit_reached)
            data["l" + str(i_level) + "_warp_delta_amt_ratio"].append(wds.ratio_above_min_threshold)
            data["l" + str(i_level) + "_warp_delta_min"].append(wds.length_min)
            data["l" + str(i_level) + "_warp_delta_max"].append(wds.length_max)
            data["l" + str(i_level) + "_warp_delta_mean"].append(wds.length_mean)
            data["l" + str(i_level) + "_warp_delta_std"].append(wds.length_standard_deviation)
            data["l" + str(i_level) + "_warp_delta_max_x"].append(wds.longest_warp_location.x)
            data["l" + str(i_level) + "_warp_delta_max_y"].append(wds.longest_warp_location.y)
            data["l" + str(i_level) + "_warps_below_min_thresh"].append(wds.is_largest_below_min_threshold)
            data["l" + str(i_level) + "_warps_above_max_thresh"].append(wds.is_largest_above_max_threshold)
            data["l" + str(i_level) + "_diff_delta_min"].append(tds.difference_min)
            data["l" + str(i_level) + "_diff_delta_max"].append(tds.difference_max)
            data["l" + str(i_level) + "_diff_delta_mean"].append(tds.difference_mean)
            data["l" + str(i_level) + "_diff_delta_std"].append(tds.difference_standard_deviation)
            data["l" + str(i_level) + "_diff_max_x"].append(tds.biggest_difference_location.x)
            data["l" + str(i_level) + "_diff_max_y"].append(tds.biggest_difference_location.y)

    return pd.DataFrame.from_dict(data)


def get_converged_ratio_for_level(data_frame, i_level):
    column_name = "l{:d}_iter_lim_reached".format(i_level)
    total_count = len(data_frame)
    grouped = data_frame.groupby(column_name)
    sizes = grouped.size()
    if False not in sizes:
        return 0.0
    else:
        return sizes[False] / total_count


def get_tsdf_difference_stats_for_level(dataframe, i_level):
    # TODO
    pass


def infer_level_count(data_frame):
    """
    :type data_frame: pandas.DataFrame
    :param data_frame:
    :return:
    """
    # assume there are 2 non-level-specific columns, 17 level-specific columns
    return (len(data_frame.columns) - 2) // 17


def analyze_convergence_data(data_frame):
    df = data_frame
    level_count = infer_level_count(df)
    print("Per-level convergence ratios:")
    for i_level in range(level_count):
        print("  level {:d}: {:.2%}".format(i_level, get_converged_ratio_for_level(data_frame, i_level)), sep="",
              end="")
    # print()
    #
    # print("Average per-level tsdf difference statistics:")
    # for i_level in range(level_count):
    #     pass


def save_bad_cases(data_frame, out_path):
    df = data_frame
    level_count = infer_level_count(df)
    unconverged_column_name = "l{:d}_iter_lim_reached".format(level_count - 1)
    max_update_x_column_name = "l{:d}_warp_delta_max_x".format(level_count - 1)
    max_update_y_column_name = "l{:d}_warp_delta_max_y".format(level_count - 1)
    bad_cases = df[['canonical_frame', 'pixel_row', max_update_x_column_name, max_update_y_column_name]] \
        [df[unconverged_column_name]]
    bad_cases.to_csv(os.path.join(out_path, "bad_cases.csv"), header=False, index=False)


def save_all_cases(data_frame, out_path):
    df = data_frame
    level_count = infer_level_count(df)
    max_update_x_column_name = "l{:d}_warp_delta_max_x".format(level_count - 1)
    max_update_y_column_name = "l{:d}_warp_delta_max_y".format(level_count - 1)
    all_cases = df[['canonical_frame', 'pixel_row', max_update_x_column_name, max_update_y_column_name]]
    all_cases.to_csv(os.path.join(out_path, "all_cases.csv"), header=False, index=False)


def get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row):
    return os.path.join(telemetry_folder, "pair_{:d}-{:d}_{:d}".format(frame_number, frame_number + 1, pixel_row))


def filter_files_based_on_case_file(case_file_path, frame_numbers_and_rows, files):
    cases = np.genfromtxt(case_file_path, delimiter=",", dtype=int)
    frame_numbers = set(cases[:, 0])
    filtered_files = []
    filtered_frame_numbers_and_rows = []
    for i_file in range(len(files)):
        frame_number, row = frame_numbers_and_rows[i_file]
        if frame_number in frame_numbers:
            filtered_files.append(files[i_file])
            filtered_frame_numbers_and_rows.append((frame_number, row))
    return filtered_files, filtered_frame_numbers_and_rows


def main():
    args = process_arguments(Arguments, "Runs 2D hierarchical optimizer on TSDF inputs generated from frame-pairs "
                                        "& random pixel rows from these. Alternatively, generates the said data or "
                                        "loads it from a folder from further re-use.")
    post_process_enum_args(args)
    perform_optimization = not Arguments.skip_optimization.v

    if args.generation_method == tsdf.GenerationMethod.EWA_TSDF_INCLUSIVE_CPP:
        generation_method_name_substring = "EWA_TI"
        generation_smoothing_substring = "_sm{:03d}".format(int(Arguments.smoothing_coefficient.v * 100))
    elif args.generation_method == tsdf.GenerationMethod.BASIC:
        generation_method_name_substring = "basic"
        generation_smoothing_substring = ""
    else:
        raise ValueError("Unsupported Generation Method")

    data_subfolder = "tsdf_pairs_128_{:s}{:s}_{:02d}".format(generation_method_name_substring,
                                                             generation_smoothing_substring,
                                                             Arguments.dataset_number.v)
    data_path = os.path.join(pu.get_reconstruction_data_directory(), "real_data/snoopy", data_subfolder)

    # TODO: add other optimizer parameters to the name
    experiment_name = "multi_{:s}_ds{:02d}_wt{:02d}_mi{:04d}_r{:02d}_ts{:02d}_ks{:02d}" \
        .format(generation_method_name_substring,
                args.dataset_number,
                int(args.max_warp_update_threshold * 100),
                args.max_iteration_count,
                int(Arguments.rate.v * 100),
                int(Arguments.tikhonov_strength.v * 100 if Arguments.tikhonov_term_enabled.v else 0),
                int(Arguments.kernel_strength.v * 100 if Arguments.gradient_kernel_enabled.v else 0)
                )

    out_path = os.path.join(args.output_path, experiment_name)
    convergence_reports_pickle_path = os.path.join(out_path, "convergence_reports.pk")

    df = None
    if not args.analyze_only:
        create_folder_if_necessary(out_path)
        if args.generate_data:
            create_or_clear_folder(data_path)
        initial_fields = []
        frame_numbers_and_rows = []
        if args.generate_data:

            datasets = esr.prepare_datasets_for_2d_frame_pair_processing(
                calibration_path=os.path.join(pu.get_reconstruction_data_directory(),
                                              "real_data/snoopy/snoopy_calib.txt"),
                frame_directory=os.path.join(pu.get_reconstruction_data_directory(),
                                             "real_data/snoopy/frames"),
                output_directory=out_path,
                y_range=(214, 400),
                replace_empty_rows=True,
                use_masks=True,
                input_case_file=Arguments.generation_case_file.v,
                offset=np.array([-64, -64, 128]),
                field_size=128,
            )

            datasets = datasets[args.start_from_index: min(len(datasets), args.stop_before_index)]

            print("Generating initial fields...")
            initial_fields_folder = os.path.join(data_path, "images")
            if args.save_initial_fields_during_generation:
                create_folder_if_necessary(initial_fields_folder)

            for dataset in progressbar.progressbar(datasets):
                canonical_field, live_field = dataset.generate_2d_sdf_fields(args.generation_method,
                                                                             args.smoothing_coefficient)
                initial_fields.append((canonical_field, live_field))
                if args.generate_data:
                    canonical_frame = infer_frame_number_from_filename(dataset.first_frame_path)
                    pixel_row = dataset.image_pixel_row
                    frame_numbers_and_rows.append((canonical_frame, pixel_row))
                    np.savez(os.path.join(data_path, "data_{:d}_{:d}".format(canonical_frame, pixel_row)),
                             canonical=canonical_field, live=live_field)
                    if args.save_initial_fields_during_generation:
                        live_frame = canonical_frame + 1
                        canonical_image_path = os.path.join(initial_fields_folder,
                                                            "tsdf_frame_{:06d}.png".format(canonical_frame))
                        viz.save_field(canonical_field, canonical_image_path, 1024 // dataset.field_size)
                        live_image_path = os.path.join(initial_fields_folder,
                                                       "tsdf_frame_{:06d}.png".format(live_frame))
                        viz.save_field(live_field, live_image_path, 1024 // dataset.field_size)

                sys.stdout.flush()
        else:
            files = os.listdir(data_path)
            print(data_path)
            files.sort()
            if files[len(files) - 1] == "images":
                files = files[:-1]
            print("Loading initial fields from {:s}...".format(data_path))
            for file in files:
                frame_numbers_and_rows.append(infer_frame_number_and_pixel_row_from_filename(file))
            if Arguments.optimization_case_file.v is not None:
                files, frame_numbers_and_rows = \
                    filter_files_based_on_case_file(Arguments.optimization_case_file.v, frame_numbers_and_rows, files)
            for file in progressbar.progressbar(files):
                archive = np.load(os.path.join(data_path, file))
                initial_fields.append((archive["canonical"], archive["live"]))

        # limit ranges
        frame_numbers_and_rows = frame_numbers_and_rows[
                                 args.start_from_index: min(len(frame_numbers_and_rows), args.stop_before_index)]
        initial_fields = initial_fields[
                         args.start_from_index: min(len(initial_fields), args.stop_before_index)]

        telemetry_logs = []
        telemetry_folder = os.path.join(out_path, "telemetry")
        if perform_optimization:
            shared_parameters = build_opt.HierarchicalOptimizer2dSharedParameters()
            shared_parameters.maximum_warp_update_threshold = args.max_warp_update_threshold
            shared_parameters.tikhonov_term_enabled = Arguments.tikhonov_term_enabled.v
            shared_parameters.gradient_kernel_enabled = Arguments.gradient_kernel_enabled.v
            shared_parameters.data_term_amplifier = Arguments.data_term_amplifier.v
            shared_parameters.tikhonov_strength = Arguments.tikhonov_strength.v
            shared_parameters.kernel = sob.generate_1d_sobolev_kernel(Arguments.kernel_size.v,
                                                                      Arguments.kernel_strength.v)
            visualization_parameters_py = build_opt.make_common_hierarchical_optimizer2d_visualization_parameters()
            logging_parameters_cpp = cpp_module.HierarchicalOptimizer2d.LoggingParameters(
                collect_per_level_convergence_reports=True,
                collect_per_level_iteration_data=args.save_telemetry

            )
            shared_parameters.maximum_iteration_count = args.max_iteration_count

            optimizer = build_opt.make_hierarchical_optimizer2d(implementation_language=args.implementation_language,
                                                                shared_parameters=shared_parameters,
                                                                logging_parameters_cpp=logging_parameters_cpp,
                                                                visualization_parameters_py=visualization_parameters_py)

            convergence_report_sets = []
            if Arguments.save_initial_and_final_fields.v or Arguments.save_telemetry.v:
                create_folder_if_necessary(telemetry_folder)

            if args.save_telemetry:
                # make all the necessary subfolders
                for frame_number, pixel_row in frame_numbers_and_rows:
                    telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                    create_folder_if_necessary(telemetry_subfolder)

            print("Optimizing...")
            i_pair = 0
            for (canonical_field, live_field) in progressbar.progressbar(initial_fields):
                (frame_number, pixel_row) = frame_numbers_and_rows[i_pair]
                live_copy = live_field.copy()
                warp_field_out = optimizer.optimize(canonical_field, live_field)
                final_live_resampled = resampling.warp_field(live_field, warp_field_out)
                if args.save_telemetry:
                    if args.implementation_language == build_opt.ImplementationLanguage.CPP:
                        telemetry_logs.append(optimizer.get_per_level_iteration_data())
                    else:
                        optimizer.visualization_parameters.out_path = \
                            get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                if Arguments.save_initial_and_final_fields.v:
                    if not args.save_telemetry:
                        frame_file_prefix = "pair_{:d}-{:d}_{:d}".format(frame_number, frame_number + 1, pixel_row)
                        final_live_path = os.path.join(telemetry_folder, frame_file_prefix + "_final_live.png")
                        canonical_path = os.path.join(telemetry_folder, frame_file_prefix + "_canonical.png")
                        initial_live_path = os.path.join(telemetry_folder, frame_file_prefix + "_initial_live.png")
                    else:
                        telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                        final_live_path = os.path.join(telemetry_subfolder, "final_live.png")
                        canonical_path = os.path.join(telemetry_subfolder, "canonical.png")
                        initial_live_path = os.path.join(telemetry_subfolder, "live.png")
                    scale = 1024 // final_live_resampled.shape[0]
                    viz.save_field(final_live_resampled, final_live_path, scale)
                    viz.save_field(canonical_field, canonical_path, scale)
                    viz.save_field(live_copy, initial_live_path, scale)

                convergence_reports = optimizer.get_per_level_convergence_reports()
                convergence_report_sets.append(convergence_reports)
                i_pair += 1

            print("Post-processing convergence reports...")
            df = post_process_convergence_report_sets(convergence_report_sets, frame_numbers_and_rows)
            df.to_excel(os.path.join(out_path, "convergence_reports.xlsx"))
            df.to_pickle(os.path.join(out_path, "convergence_reports.pk"))

        if Arguments.save_telemetry.v and \
                Arguments.implementation_language.v == build_opt.ImplementationLanguage.CPP and \
                len(telemetry_logs) > 0:
            print("Saving C++-based telemetry (" + telemetry_folder + ")...")
            i_pair = 0
            telemetry_metadata = ho_viz.get_telemetry_metadata(telemetry_logs[0])
            for telemetry_log in progressbar.progressbar(telemetry_logs):
                (frame_number, pixel_row) = frame_numbers_and_rows[i_pair]
                telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                ho_viz.save_telemetry_log(telemetry_log, telemetry_metadata, telemetry_subfolder)
                i_pair += 1

        if Arguments.convert_telemetry.v and \
                Arguments.implementation_language.v == build_opt.ImplementationLanguage.CPP:
            # TODO: attempt to load telemetry if the array is empty
            if len(telemetry_logs) == 0:
                print("Loading C++-based telemetry (" + telemetry_folder + ")...")
                for frame_number, pixel_row in progressbar.progressbar(frame_numbers_and_rows):
                    telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                    telemetry_log = ho_viz.load_telemetry_log(telemetry_subfolder)
                    telemetry_logs.append(telemetry_log)

            print("Converting C++-based telemetry to videos (" + telemetry_folder + ")...")
            i_pair = 0
            total_frame_count = ho_viz.get_number_of_frames_to_save_from_telemetry_logs(telemetry_logs)
            bar = progressbar.ProgressBar(max_value=total_frame_count)
            telemetry_metadata = ho_viz.get_telemetry_metadata(telemetry_logs[0])
            for telemetry_log in telemetry_logs:
                canonical_field, live_field = initial_fields[i_pair]
                (frame_number, pixel_row) = frame_numbers_and_rows[i_pair]
                telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                ho_viz.convert_cpp_telemetry_logs_to_video(telemetry_log, telemetry_metadata,
                                                           canonical_field, live_field, telemetry_subfolder, bar)
                i_pair += 1

    else:
        df = pd.read_pickle(convergence_reports_pickle_path)

    if df is not None:
        analyze_convergence_data(df)
        if not Arguments.optimization_case_file.v:
            save_bad_cases(df, out_path)
            save_all_cases(df, out_path)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
