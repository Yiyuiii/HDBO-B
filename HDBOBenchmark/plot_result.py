import re
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from .utils.logging import logger
from HDBOBenchmark.utils.plot import (
    get_linestyle_list,
    cal_std_data,
    plt_init,
    sub_fig,
    plot,
)
from HDBOBenchmark.utils.os import load_pickle

wilcoxon_method = "ALEBO"


def plot_result(func_list, method_list, result_path="./result"):

    max_len = 100

    # init
    plt_init(style="seaborn-paper")  # bmh for common, seaborn-paper for paper
    linestyle_list = get_linestyle_list(len(method_list))

    # cons
    save_dir = "./figs/result/"

    # Params = ({'ymin': -50, 'ymax': 100000},
    #           {},
    #           {'ymin': -200, 'ymax': 1500})
    Params = ({}, {}, {})
    # excel
    excel_data = dict()
    excel_frames = []  # [30, 50, 100]
    for excel_i in excel_frames:
        excel_data[str(excel_i)] = {"func": list()}
        for method in method_list:
            excel_data[str(excel_i)][method] = list()
            excel_data[str(excel_i)][method + "_std"] = list()
    excel_data["wilcoxon"] = {"func": list()}
    for method in method_list:
        excel_data["wilcoxon"][method] = list()

    method_rank_data = {method: [] for method in method_list}
    # loop
    if not isinstance(func_list, list):
        func_list = [func_list]
    for i, func in enumerate(func_list):
        name = func.name
        dim = func.dim
        fig = plt.figure()
        ax = sub_fig(fig, xlabel="Iteration", ylabel="Historical Best Value")
        excel_data["wilcoxon"]["func"].append(name)
        for excel_i in excel_frames:
            excel_data[str(excel_i)]["func"].append(name)

        wilcoxon_data = dict()
        data_max_step = -1
        method_index = 0
        method_data = {method: [] for method in method_list}
        for method in method_list:
            excel_data["wilcoxon"][method].append("N/A")
            for excel_i in excel_frames:
                excel_data[str(excel_i)][method].append("N/A")
                excel_data[str(excel_i)][method + "_std"].append("N/A")
            wilcoxon_data[method] = list()
            load_cnt = 0
            load_path = os.path.join(
                result_path, f"{method}/{name}_{dim}_{load_cnt}.pkl"
            )
            data = list()
            while os.path.exists(load_path):
                result_dict = load_pickle(load_path)[0]
                if isinstance(result_dict, dict):
                    if "best_y_history" in result_dict.keys():
                        best_y_history = result_dict["best_y_history"]
                    elif "y_history" in result_dict.keys():
                        best_y_history = result_dict["y_history"]
                        _y = np.inf
                        for k in range(len(best_y_history)):
                            if best_y_history[k] > _y:
                                best_y_history[k] = _y
                            _y = best_y_history[k]
                            best_y_history[k] = min(best_y_history[k], 100 + 10)
                    else:
                        logger.warning(f"{load_path} doesnt provide a valid dict.")
                    if len(best_y_history) > 100:
                        best_y_history = best_y_history[:max_len]
                    data.append(
                        {
                            "x": np.arange(1, len(best_y_history) + 1),
                            "y": best_y_history,
                            "name": method,
                        }
                    )
                    method_data[method].append(best_y_history)
                    data_max_step = max(data_max_step, len(best_y_history))
                else:
                    logger.warning(f"{load_path} is not in valid format.")
                load_cnt += 1
                load_path = os.path.join(
                    result_path, f"{method}/{name}_{dim}_{load_cnt}.pkl"
                )
            if len(data) == 0:  # no data
                continue
            for d in data:
                wilcoxon_data[method].append(float(d["y"][-1]))
            data = cal_std_data(*data)
            plot(
                data,
                ax,
                color=linestyle_list[method_index][0],
                linestyle=linestyle_list[method_index][1],
            )
            # save excel
            for excel_i in excel_frames:
                excel_data[str(excel_i)][method][-1] = data["y"][excel_i - 1]
                excel_data[str(excel_i)][method + "_std"][-1] = data["std"][excel_i - 1]
            method_index += 1

        if data_max_step == -1:
            continue

        oracle_value = func.min_val
        ax.plot(
            np.arange(data_max_step + 1),
            oracle_value * np.ones(data_max_step + 1),
            f"r-.",
            alpha=0.7,
        )
        if i < len(Params) and "ymin" in Params[i]:
            ax.set_ylim(ymin=Params[i]["ymin"], ymax=Params[i]["ymax"])
            name += "_local"

        # For paper
        ax.set_ylim(ymin=-5, ymax=105)
        # ax.set_yscale("log", base=10)
        title = name
        # name_pre, name_suf = re.match("([0-9]*)\.(.*)", name).groups()
        # title = "No." + name_pre + name_suf
        # if len(title) >= 72:
        #     title = title[:70] + "\n" + title[70:]
        ax.set_title(title, loc="center")

        ax.legend(loc="best", frameon=True, facecolor="white")
        plt.tight_layout()

        # write png to file
        os.makedirs(save_dir, exist_ok=True)
        save_img_path = os.path.join(save_dir, f"{name}.png")
        if os.path.exists(save_img_path):
            os.remove(save_img_path)
        plt.savefig(save_img_path, bbox_inches="tight", dpi=300)
        logger.info(f"Img saved at {save_img_path}.")
        plt.close(fig)

        # Rank calculation
        rank_data = {method: [] for method in method_list}
        for method in method_list:
            if len(method_data[method]) == 0:
                method_data.pop(method)
                method_list.remove(method)
        min_traj_length = min(len(trajs) for trajs in method_data.values())
        for i_traj in range(min_traj_length):
            traj_values = {method: [] for method in method_list}
            for step in range(data_max_step):
                step_values = []
                for method in method_list:
                    if step < len(method_data[method][i_traj]):
                        step_values.append((method, method_data[method][i_traj][step]))
                    else:
                        step_values.append((method, method_data[method][i_traj][-1]))
                step_values.sort(key=lambda x: x[1])
                for rank, (method, _) in enumerate(step_values):
                    traj_values[method].append(rank + 1)
            for method in method_list:
                rank_data[method].append(
                    {
                        "x": np.arange(1, len(traj_values[method]) + 1),
                        "y": traj_values[method],
                        "name": method,
                    }
                )

        plot_rank_data(
            method_list=method_list,
            rank_data=rank_data,
            save_dir=save_dir,
            fig_name=name,
        )
        for method in method_list:
            method_rank_data[method].extend(rank_data[method])

        # write excel
        # for excel_name in excel_data.keys():
        #     df = pd.DataFrame(excel_data[excel_name])
        #     save_excel_path = os.path.join(save_dir, f"data_{excel_name}.csv")
        #     if os.path.exists(save_excel_path):
        #         os.remove(save_excel_path)
        #     df.to_csv(save_excel_path, float_format="%.2f")

    plot_rank_data(
        method_list=method_list,
        rank_data=method_rank_data,
        save_dir=save_dir,
        fig_name="all",
    )


def plot_rank_data(method_list, rank_data, save_dir, fig_name):
    linestyle_list = get_linestyle_list(len(method_list))
    # Plot rank data
    fig = plt.figure()
    ax = sub_fig(fig, xlabel="Iteration", ylabel="Rank")
    method_index = 0
    for method in method_list:
        if len(rank_data[method]) == 0:  # no data
            continue
        data = cal_std_data(*rank_data[method])
        plot(
            data,
            ax,
            color=linestyle_list[method_index][0],
            linestyle=linestyle_list[method_index][1],
        )
        method_index += 1
    ax.legend(loc="best", frameon=True, facecolor="white")
    plt.tight_layout()

    # Save rank plot
    save_rank_path = os.path.join(save_dir, f"{fig_name}_rank.png")
    if os.path.exists(save_rank_path):
        os.remove(save_rank_path)
    plt.savefig(save_rank_path, bbox_inches="tight", dpi=300)
    logger.info(f"Rank img saved at {save_rank_path}.")
    plt.close(fig)


def move_result(cur_dir, target_dir):
    import shutil
    import os

    file_list = os.listdir(cur_dir)
    for f in file_list:
        result = re.match("(.*_)[0-9]*(\.pkl)", f)
        if not result:
            continue
        f_pre, f_suf = result.groups()
        n = 0
        f_new = os.path.join(target_dir, f"{f_pre}{n}{f_suf}")
        while os.path.exists(f_new):
            n += 1
            f_new = os.path.join(target_dir, f"{f_pre}{n}{f_suf}")
        shutil.move(os.path.join(cur_dir, f), f_new)
