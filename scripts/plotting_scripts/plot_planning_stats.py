import os
import numpy as np
import matplotlib.pyplot as plt
from autolab_core import YamlConfig
import argparse

task_names = ["RodInBox", "RodInDrawer"]
task_to_num_models = {"RodInBox":2, "RodInDrawer":3}
task_to_num_skills = {"RodInBox":2, "RodInDrawer":3}
FONT_SIZE=13
default_fontdict={'size':FONT_SIZE}

def average_data_across_runs(plan_results_dir, task, dir_prefix):
    if plan_results_dir is None:
        return None
    full_filepath = os.path.join(dir_prefix, plan_results_dir, "plan_results")

    combined_data = {'elapsed_time':[],
                     'num_model_evals':[],
                     'model_evals_per_skill' :{}
                     }
    for skill_idx in range(task_to_num_skills[task]):
        combined_data["model_evals_per_skill"][skill_idx] = {model_idx: [] for model_idx in range(task_to_num_models[task])}
    for fn in os.listdir(full_filepath):
        if "error" in fn:
            continue #Something wrong haproded during this plan, likely a forgotten reset
        datalist = np.load(os.path.join(full_filepath, fn), allow_pickle=True)
        for data in datalist:
            combined_data["elapsed_time"].append(data["elapsed_time"])
            for skill_idx in range(task_to_num_skills[task]):
                combined_data["num_model_evals"].append(sum([sum(value) for value in data['model_type_per_skill_idx'].values()]))
                #print(data["num_model_evals"])
                for model_idx in range(task_to_num_models[task]):
                    try:
                        combined_data['model_evals_per_skill'][skill_idx][model_idx].append(data["model_type_per_skill_idx"][skill_idx][model_idx])
                    except IndexError:
                        continue

    combined_data["elapsed_time"] = np.mean(combined_data["elapsed_time"])
    combined_data["num_model_evals"] = np.mean(combined_data["num_model_evals"])
    for skill_idx in range(task_to_num_skills[task]):
        for model_idx in range(task_to_num_models[task]):
            try:
                combined_data['model_evals_per_skill'][skill_idx][model_idx] = np.mean( combined_data['model_evals_per_skill'][skill_idx][model_idx])
            except:
                continue
    return combined_data

def plot_bar_graph_stats(combined_data_for_method, task, max_y, ax, label_x = 0, label_y=0,title=""):
    ind = np.arange(task_to_num_skills[task])
    width = 0.5
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    model_uses_per_skill_per_model = []
    if label_x:
        ax.set_xlabel("Skill", fontdict=default_fontdict)
    if label_y:
        ax.set_ylabel("Model evaluations", fontdict=default_fontdict)
    ax.set_title(title, fontdict=default_fontdict)
    if task == "RodInDrawer":
        model_to_model_names=["Simulator", "Analytical (Drawer)", "Analytical \n (Pick & Place)"]
    else:
        model_to_model_names = ["Simulator", "Analytical\n (Pick & Place)"]
    ax.set_ylim([0,max_y])
    colors = ["blue", "orange", "purple"]
    for model_idx in range(task_to_num_models[task]):
        model_uses_per_skill = np.zeros((task_to_num_skills[task],))
        for skill_idx in range(task_to_num_skills[task]):
            model_uses_per_skill[skill_idx] = combined_data_for_method['model_evals_per_skill'][skill_idx][model_idx]
        if model_idx == 0:
            ax.bar(ind, model_uses_per_skill, width=width, label = model_to_model_names[model_idx])
        else:
            sum_bottoms = np.sum(np.vstack(model_uses_per_skill_per_model), axis=0)
            ax.bar(ind, model_uses_per_skill, width=width, bottom=sum_bottoms, label = model_to_model_names[model_idx], color=colors[model_idx])
        model_uses_per_skill_per_model.append(model_uses_per_skill)
    ax.set_xticks(ind)
    skill_font_dict = {'style': 'italic', 'size': FONT_SIZE}
    if task == "RodInDrawer":
        ax.set_xticklabels(["OpenDrawer", "Pick", "LiftAndDrop"], fontdict=skill_font_dict)
    else:
        ax.set_xticklabels(["Pick", "LiftAndDrop"], fontdict=skill_font_dict)

def main(args):
    cfg = YamlConfig(args.cfg)
    time_methods = cfg["time_methods"]
    plot_methods = cfg["plot_methods"]

    dir_prefix = cfg["dir_prefix"]

    method_to_task_to_avg_data = {}
    for method in time_methods:
        method_to_task_to_avg_data[method] = {}
        for task in task_names:
            if task not in cfg["per_method_info"][method].keys():
                continue
            average_data = average_data_across_runs(cfg["per_method_info"][method][task]['plan_results_dir'], task, dir_prefix)
            if average_data is None:
                continue
            print(method)
            print("Task", task)
            print(f" Average elapsed time: {average_data['elapsed_time']}")
            print(f" Num model evals: {average_data['num_model_evals']}")
            print(f" Model eval rate: {average_data['num_model_evals']/average_data['elapsed_time']}")
            print("\n ")
            method_to_task_to_avg_data[method][task] = average_data
    max_num_models_eval = -np.inf
    task_for_bar_graph = cfg["task_for_bar_graph"]
    for method in plot_methods:
        average_data = method_to_task_to_avg_data[method][task_for_bar_graph]
        max_num_models_eval = max(max_num_models_eval,
                                  max([sum(average_data["model_evals_per_skill"][skill_idx].values()) for skill_idx in range(task_to_num_skills[task_for_bar_graph])])
                                  )
    fig, axes = plt.subplots(1,3)
    for method_idx, method in enumerate(plot_methods):
        plot_bar_graph_stats(method_to_task_to_avg_data[method][task_for_bar_graph], task_for_bar_graph, max_num_models_eval,
                             axes[method_idx],
                             label_x = method_idx ==1, #the middle one
                             label_y = method_idx == 0,
                             title=cfg["per_method_info"][method]["title"])
    if task_for_bar_graph == "RodInDrawer":
        plt.legend(prop=default_fontdict)
    plt.subplots_adjust(bottom=0.7)
    plt.subplots_adjust(left=0.4)
    if task_for_bar_graph == "RodInDrawer":
        plt.subplots_adjust(wspace=0.17)
    else:
        plt.subplots_adjust(wspace=0.1)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", '-c', type=str, default="cfg/plot_planner_data.yaml")
    args = parser.parse_args()
    main(args)
