import datetime as dt
import itertools as it
import math
import os
import time as t
import argparse
import numpy as np
import pandapower as pp
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial, reduce


# function definitions #########################################################
def load_data(base_path):
    print(f"Loading grid data..")
    net = pp.from_pickle(os.path.join(base_path, "cossmic_grid.p"))
    loads = pd.read_table(
        os.path.join(
            base_path,
            "..",
            "0_data preparation",
            "Cossmic_grid",
            "grid-data",
            "loads.csv",
        ),
        sep=",",
    )
    allowed_sw = np.load(os.path.join(base_path, "allowed_switches.npy"))

    return net, loads, allowed_sw


def gen_combinations(net, loads, allowed_sw, seed):
    print(f"Generating topology combinations..")
    state_lists = []
    for _, row in loads.iterrows():
        states = [row["p_max"], row["p_set"], -row["p_max"], row["p_max"] * 3]
        state_lists.append(states)

    all_combinations = list(it.product(*state_lists))

    df_combinations = pd.DataFrame(all_combinations, columns=loads.name)

    original_list = [False] * len(loads.name)

    index_combinations = list(it.combinations(range(len(loads.name)), 3))

    modified_lists = []

    for indices in index_combinations:
        modified_list = original_list.copy()

        for index in indices:
            modified_list[index] = True

        modified_lists.append(modified_list)

    t0 = t.time()
    result_list = []

    for comb_index in df_combinations.index:
        this_comb = df_combinations.loc[comb_index].copy()

        for modified_list in modified_lists:
            modified_comb_1 = this_comb.copy()
            modified_comb_2 = this_comb.copy()
            for index, element in enumerate(modified_list):
                if element:
                    modified_comb_1.iloc[index] = loads.p_max.iloc[index] * 3
                    modified_comb_2.iloc[index] = -loads.p_max.iloc[index]
            result_list.append(modified_comb_1)
            result_list.append(modified_comb_2)

    all_results_df = pd.DataFrame(result_list, columns=loads.name)
    all_results_df.reset_index(drop=True, inplace=True)
    all_results_df = all_results_df.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )
    ordered_columns = net.load["name"].tolist()
    all_results_df = all_results_df[ordered_columns]
    t1 = t.time()
    t_elapsed = t1 - t0
    print("Time elapsed: %.1f sec." % t_elapsed)

    return all_results_df


def identify_problematic_situations(result_df, net):
    print(f"Identifying problematic situations..")
    
    t0 = t.time()

    Voltage_mat = np.empty(
        shape=[0, len(net.bus[net.bus["name"].isin(net.load["name"])])]
    )
    Load_mat = np.empty(shape=[0, len(net.load.p_mw)])
    Problem_num = 0
    Problem_detail_mat = np.empty(
        shape=[0, len(net.bus) + len(net.trafo) + len(net.line)]
    )

    notconverged = 0

    ordered_columns = net.load["name"].tolist()
    result_df = result_df[ordered_columns]

    for timestamp, row in result_df.iterrows():
        net.load["p_mw"] = row.values

        try:
            pp.runpp(net, numba=True)

            Trafo_overload = np.asarray(net.res_trafo.loading_percent > 100)
            Bus_voltage_offspec = np.asarray(
                (net.res_bus.vm_pu > 1.03) | (net.res_bus.vm_pu < 0.97)
            )
            Line_overload = np.asarray(net.res_line.loading_percent > 100)
            PB_V = np.concatenate((Trafo_overload, Bus_voltage_offspec, Line_overload))

            Problem = np.sum(PB_V) > 0

            if Problem:
                Voltage_mat = np.append(
                    Voltage_mat,
                    [net.res_bus.vm_pu[net.bus["name"].isin(net.load["name"])]],
                    axis=0,
                )
                Load_mat = np.append(Load_mat, [net.load.p_mw.values], axis=0)
                Problem_detail_mat = np.append(Problem_detail_mat, PB_V)
                Problem_num += 1
        except:
            notconverged += 1

    t1 = t.time()
    execution_time_minutes = (t1 - t0) / 60
    print(f"Total execution time: {execution_time_minutes:.2f} minutes")
    print("There are", Problem_num, "problematic cases out of", len(result_df))
    print("There are ", notconverged, "cases in which the simulation did not converge")

    return Voltage_mat, Load_mat, Problem_detail_mat


def find_solution(net, allowed_sw, Load_mat):
    base_switch = net.switch.closed.loc[net.switch.et == "l"]
    base_trafo = net.trafo.tap_pos

    impossible_c = 0

    #t0 = t.time()
    possible_c = 0
    notconverged = 0

    tap_pos_ranges = [
        range(row["tap_min"], row["tap_max"] + 1) for index, row in net.trafo.iterrows()
    ]
    op_length = len(net.switch[net.switch.et == "l"]) + len(net.trafo) + 1
    max_changes = len(net.switch[net.switch.et == "l"]) + sum(
        abs(net.trafo.tap_min - net.trafo.tap_max)
    )

    min_chg_from_base_vector = np.empty(shape=[0, op_length], dtype=int)
    min_dev_from_norm_V_vector = np.empty(shape=[0, op_length], dtype=int)

    for d in tqdm(range(len(Load_mat))):
        min_dev_from_norm_V = np.inf
        min_dev_from_norm_V_conf = np.empty(shape=[op_length], dtype=int)
        min_dev_from_norm_pb_V = np.inf
        min_dev_from_norm_pb_V_conf = np.empty(shape=[op_length], dtype=int)
        min_chg_from_base = max_changes
        min_chg_from_base_conf = np.empty(shape=[op_length], dtype=int)

        no_pb_cnt = 0
        notconverged = 0

        net.load["p_mw"] = Load_mat[d]

        for switches in range(len(allowed_sw)):
            net.switch.loc[net.switch.et == "l", "closed"] = allowed_sw[switches]

            supplied = len(pp.topology.unsupplied_buses(net)) == 0
            if supplied == False:
                print(
                    "The following buses are unsupplied:",
                    pp.topology.unsupplied_buses(net),
                )

            for combination in it.product(*tap_pos_ranges):
                net.trafo.tap_pos = combination

                try:
                    pp.runpp(net, numba=True)

                    Trafo_overload = np.asarray(net.res_trafo.loading_percent > 100)
                    Bus_voltage_offspec = np.asarray(
                        (net.res_bus.vm_pu > 1.03) | (net.res_bus.vm_pu < 0.97)
                    )
                    Line_overload = np.asarray(net.res_line.loading_percent > 100)

                    PB_V = np.concatenate(
                        (Trafo_overload, Bus_voltage_offspec, Line_overload)
                    )
                    no_problem = np.sum(PB_V) == 0

                    dev_from_norm_V = np.sum((net.res_bus.vm_pu - 1) ** 2)
                    chg_from_base = np.sum(base_switch ^ allowed_sw[switches]) + np.sum(
                        base_trafo ^ net.trafo.tap_pos.values
                    )

                    if dev_from_norm_V < min_dev_from_norm_pb_V:
                        min_dev_from_norm_pb_V = dev_from_norm_V
                        min_dev_from_norm_pb_V_conf = np.append(
                            np.append(
                                allowed_sw[switches] * 1,
                                net.trafo.tap_pos.values.tolist(),
                            ),
                            0,
                        )

                    if no_problem:
                        no_pb_cnt += 1

                        if dev_from_norm_V < min_dev_from_norm_V:
                            min_dev_from_norm_V = dev_from_norm_V
                            min_dev_from_norm_V_conf = np.append(
                                np.append(
                                    allowed_sw[switches] * 1,
                                    net.trafo.tap_pos.values.tolist(),
                                ),
                                1,
                            )

                        if chg_from_base < min_chg_from_base:
                            min_chg_from_base = chg_from_base

                            min_chg_from_base_conf = np.append(
                                np.append(
                                    allowed_sw[switches] * 1,
                                    net.trafo.tap_pos.values.tolist(),
                                ),
                                1,
                            )

                except:
                    notconverged += 1

        if no_pb_cnt > 0:
            possible_c += 1

            min_chg_from_base_vector = np.append(
                min_chg_from_base_vector, [min_chg_from_base_conf], axis=0
            )
            min_dev_from_norm_V_vector = np.append(
                min_dev_from_norm_V_vector, [min_dev_from_norm_V_conf], axis=0
            )

        else:
            impossible_c += 1

            min_chg_from_base_vector = np.append(
                min_chg_from_base_vector, [min_dev_from_norm_pb_V_conf], axis=0
            )
            min_dev_from_norm_V_vector = np.append(
                min_dev_from_norm_V_vector, [min_dev_from_norm_pb_V_conf], axis=0
            )

    print(
        "-------------------------------------------------------------------------------"
    )
    print("We analyzed", len(Load_mat), "problematic load configurations:")
    print(
        "There are",
        possible_c,
        "possible cases with a solution ",
        100 * possible_c / (d + 1),
        "%",
    )
    print(
        "There are",
        impossible_c,
        "impossible cases (without a solution) ",
        100 * impossible_c / (d + 1),
        "%",
    )
    print(
        "Number of possible switching states:",
        len(allowed_sw) * math.prod(abs(net.trafo.tap_min - net.trafo.tap_max) + 1),
    )

    #t1 = t.time()
    #execution_time_minutes = (t1 - t0) / 60
    #print(f"Total execution time: {execution_time_minutes:.2f} minutes")

    return min_dev_from_norm_V_vector, min_chg_from_base_vector


def gen_train_data(result_df, net, allowed_sw, seed):
    Voltage_mat, Load_mat, Problem_detail_mat = identify_problematic_situations(
        result_df, net
    )
    min_dev_from_norm_V_vector, min_chg_from_base_vector = find_solution(
        net, allowed_sw, Load_mat
    )
    return (
        Voltage_mat,
        Load_mat,
        Problem_detail_mat,
        min_dev_from_norm_V_vector,
        min_chg_from_base_vector,
    )

def divide_dataframe_into_lists(df, num_divisions):
    """
    Divides a DataFrame into a specified number of smaller lists of lists,
    each containing an equal number of rows from the DataFrame.
    
    Parameters:
    - df: The pandas DataFrame to be divided.
    - num_divisions: The number of divisions or lists of lists to create.
    
    Returns:
    - A list of lists, where each inner list contains rows from the DataFrame.
    """
    # Calculate the size of each division
    division_size = len(df) // num_divisions
    
    # Initialize an empty list to store the smaller lists
    divided_lists = []
    
    # Slice the DataFrame and add to the list
    for i in range(num_divisions):
        start_index = i * division_size
        # For the last division, include any remaining rows
        if i == num_divisions - 1:
            end_index = len(df)
        else:
            end_index = (i + 1) * division_size
        
        # Convert DataFrame slice to a list of lists and append to our list
        divided_lists.append(df.iloc[start_index:end_index].values.tolist())
    
    return divided_lists

def main(base_path, results_path, max_workers):
    strTime = dt.datetime.fromtimestamp(t.time()).strftime("%Y-%m-%d_%H-%M-%S")
    seed=42
    net, loads, allowed_sw = load_data(base_path)

    if os.path.exists(os.path.join('data', "raw", str(seed) + "_time_series_data.feather")):
        print(f"Loading topology combinations..")
        all_results_df = pd.read_feather(os.path.join(results_path, "raw", str(seed) + "_time_series_data.feather"))
    else: 
        all_results_df = gen_combinations(net, loads, allowed_sw, seed)
        all_results_df.to_feather(
            os.path.join(results_path, "raw", str(seed) + "_time_series_data.feather")
        )
    
    fn = partial(gen_train_data, net=net, allowed_sw=allowed_sw, seed=seed)
    
    #ToDo: Pass as input value with , check whether num can be divided by num_workers, enter #values/num_workers in list_of_chunks
    result_df = all_results_df[93000:120000]
    
    list_of_chunks = [result_df[i:i+4500] for i in range(0, len(result_df), 4500)]
    print(f"Finding solutions..")
    t0 = t.time()
    results = process_map(
        fn, list_of_chunks, max_workers= max_workers, desc="Searching Solutions"
    )
    (
        Voltage_mat,
        Load_mat,
        Problem_detail_mat,
        min_dev_from_norm_V_vector,
        min_chg_from_base_vector,
    ) = tuple(map(lambda x: np.concatenate(x, axis = 0), zip(*results)))

    # save the data
    np.save(
        os.path.join(
            results_path, "preprocessed", strTime + "_voltage_distribution.npy"
        ),
        Voltage_mat,
    )
    np.save(
        os.path.join(
            results_path, "preprocessed", strTime + "_Load_distribution.npy"
        ),
        Load_mat,
    )
    np.save(
        os.path.join(results_path, "preprocessed", strTime + "_Problem_details.npy"),
        Problem_detail_mat,
    )

    filename_chg = "_min_chg_from_base.npy"
    filename_dev = "_min_dev_from_norm_V.npy"
    np.save(
        os.path.join(results_path, "preprocessed", strTime + "_" + filename_dev),
        min_dev_from_norm_V_vector,
    )
    np.save(
        os.path.join(results_path, "preprocessed", strTime + "_" + filename_chg),
        min_chg_from_base_vector,
    )

    t1 = t.time()
    execution_time_minutes = (t1-t0)/60
    print(f"Total execution time: {execution_time_minutes:.2f} minutes")
    
# main #########################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate train data for CNN")

    parser.add_argument(
        "base_path",
        type=str,
        help="path of input data",
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="path to save results",
    )
    parser.add_argument(
        "--max_workers", required=True, type=int, help="number of parallel processes to execute"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(base_path=args.base_path, results_path=args.results_path, max_workers=args.max_workers)
