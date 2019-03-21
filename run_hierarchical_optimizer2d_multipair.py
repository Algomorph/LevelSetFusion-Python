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
import experiment.build_hierarchical_optimizer_helper as build_opt
import utils.visualization as viz
import nonrigid_opt.hierarchical.hierarchical_optimization_visualizer as ho_viz
from experiment.multipair_hierarchical_optimizer_arguments import Arguments, legacy_process_args, post_process_enum_args
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
    print()

    print("Average per-level tsdf difference statistics:")
    for i_level in range(level_count):
        pass


def get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row):
    return os.path.join(telemetry_folder, "pair_{:d}-{:d}_{:d}".format(frame_number, frame_number + 1, pixel_row))


def main():
    args = process_arguments(Arguments, "Runs 2D hierarchical optimizer on TSDF inputs generated from frame-pairs "
                                        "& random pixel rows from these. Alternatively, generates the said data or "
                                        "loads it from a folder from further re-use.")
    post_process_enum_args(args)

    perform_optimization = not args.skip_optimization
    load_data = not args.generate_data and perform_optimization

    if args.generation_method == tsdf.GenerationMethod.EWA_TSDF_INCLUSIVE_CPP:
        method_name_substring = "TSDF_inclusive"
    elif args.generation_method == tsdf.GenerationMethod.BASIC:
        method_name_substring = "basic"
    else:
        raise ValueError("Unsupported Generation Method")

    experiment_name = "multi_{:s}_{:d}_{:d}_{:02d}".format(method_name_substring,
                                                           int(args.max_warp_update_threshold * 100),
                                                           args.max_iteration_count, args.dataset_number)
    data_subfolder = "tsdf_pairs_128_{:s}_{:02d}".format(method_name_substring, args.dataset_number)
    out_path = os.path.join(args.output_path, experiment_name)
    convergence_reports_pickle_path = os.path.join(out_path, "convergence_reports.pk")
    data_path = os.path.join(pu.get_reconstruction_directory(), "real_data/snoopy", data_subfolder)

    df = None
    if not args.analyze_only:
        create_folder_if_necessary(out_path)
        if args.generate_data:
            create_or_clear_folder(data_path)
        initial_fields = []
        frame_numbers_and_rows = []
        if not load_data or args.generate_data:
            datasets = esr.prepare_datasets_for_2d_frame_pair_processing(
                calibration_path=os.path.join(pu.get_reconstruction_directory(),
                                              "real_data/snoopy/snoopy_calib.txt"),
                frame_directory=os.path.join(pu.get_reconstruction_directory(),
                                             "real_data/snoopy/frames"),
                output_directory=out_path,
                y_range=(214, 400),
                replace_empty_rows=True,
                use_masks=True,
                input_case_file=None,
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
        if load_data:
            files = os.listdir(data_path)
            print(data_path)
            files.sort()
            if files[len(files) - 1] == "images":
                files = files[:-1]
            print("Loading initial fields from {:s}...".format(data_path))

            for file in progressbar.progressbar(files):
                frame_numbers_and_rows.append(infer_frame_number_and_pixel_row_from_filename(file))
                archive = np.load(os.path.join(data_path, file))
                initial_fields.append((archive["canonical"], archive["live"]))

        # limit ranges
        frame_numbers_and_rows = frame_numbers_and_rows[
                                 args.start_from_index: min(len(frame_numbers_and_rows), args.stop_before_index)]
        initial_fields = initial_fields[
                         args.start_from_index: min(len(initial_fields), args.stop_before_index)]

        if perform_optimization:
            shared_parameters = build_opt.HierarchicalOptimizer2dSharedParameters()
            shared_parameters.maximum_warp_update_threshold = 0.01
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
            telemetry_folder = os.path.join(out_path, "telemetry")
            if args.save_final_fields or args.save_telemetry:
                create_folder_if_necessary(telemetry_folder)

            if args.save_telemetry:
                # make all the necessary subfolders
                for frame_number, pixel_row in frame_numbers_and_rows:
                    telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                    create_or_clear_folder(telemetry_subfolder)

            print("Optimizing...")
            i_pair = 0
            telemetry_logs = []
            for (canonical_field, live_field) in progressbar.progressbar(initial_fields):
                (frame_number, pixel_row) = frame_numbers_and_rows[i_pair]
                warp_field_out = optimizer.optimize(canonical_field, live_field)
                final_live_resampled = resampling.warp_field(live_field, warp_field_out)
                if args.save_telemetry:
                    if args.implementation_language == build_opt.ImplementationLanguage.CPP:
                        telemetry_logs.append(optimizer.get_per_level_iteration_data())
                    else:
                        optimizer.visualization_parameters.out_path = \
                            get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                if args.save_final_fields:

                    if not args.save_telemetry:
                        final_live_path = os.path.join(telemetry_folder,
                                                       "pair_{:d}-{:d}_{:d}_final_live.png".format(frame_number,
                                                                                                   frame_number + 1,
                                                                                                   pixel_row))
                    else:
                        telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                        final_live_path = os.path.join(telemetry_subfolder, "live_warped.png")
                    viz.save_field(final_live_resampled, final_live_path, 1024 // final_live_resampled.shape[0])

                convergence_reports = optimizer.get_per_level_convergence_reports()
                convergence_report_sets.append(convergence_reports)
                i_pair += 1

            if args.save_telemetry and args.implementation_language == build_opt.ImplementationLanguage.CPP:
                print("Converting telemetry to videos (" + telemetry_folder + ")...")
                i_pair = 0
                for telemetry_log in progressbar.progressbar(telemetry_logs):
                    canonical_field, live_field = initial_fields[i_pair]
                    (frame_number, pixel_row) = frame_numbers_and_rows[i_pair]
                    telemetry_subfolder = get_telemetry_subfolder_path(telemetry_folder, frame_number, pixel_row)
                    ho_viz.convert_cpp_optimization_iteration_data_to_video(telemetry_log, canonical_field, live_field,
                                                                            telemetry_subfolder)
                    i_pair += 1

            print("Post-processing convergence reports...")
            df = post_process_convergence_report_sets(convergence_report_sets, frame_numbers_and_rows)
            df.to_excel(os.path.join(out_path, "convergence_reports.xlsx"))
            df.to_pickle(os.path.join(out_path, "convergence_reports.pk"))
    else:
        df = pd.read_pickle(convergence_reports_pickle_path)

    if df is not None:
        analyze_convergence_data(df)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
