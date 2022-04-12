import matplotlib.pyplot as plt
import numpy as np
from autolab_core import YamlConfig
plt.rcParams['font.size']=21#35
data_type = "plan_success"
MONOSPACE_FONT_DICT = {"family":"monospace", "weight": "bold"}
task_names = ["BoxInLocation", "WaterInBox"]
#methods = [anchor_baseline, random_baseline, simulator_only, analytical_only_rod_and_robot, analytical_only_rod_and_drawer, ours]

plt.xlabel("Task")
plt.ylabel("Success rate")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cfg = {'bar_width' : 0.25, 'bar_spacing':0.01}
method_names = ["ours", "control"]
if data_type == "plan_success":
    data = {
            #after new controllers, 25/48 ours, 13/50 baseline
            "control":[24/50, 13/50.],
            "ours":[37/38, 25/48.]

            }
else:
    data = {
            "control":[50/50., 50/50.],
            "ours":[38/50.,48/50.]

            }
len_bars_per_task = 0.2
for i, method_name in enumerate(method_names):
    mean_scores = data[method_name]
    X = np.arange(2)
    rects = ax.bar(X + i*(cfg["bar_width"]+cfg["bar_spacing"]), mean_scores, width=cfg["bar_width"], label=method_name)
    #ax.bar_label(ax.containers[i], fmt="%.2f")
    len_bars_for_task  = len(method_names)*(cfg['bar_width'] + cfg["bar_spacing"])
    ax.set_xticks([x + len_bars_for_task/2 - cfg['bar_width']/2  - cfg["bar_spacing"]/2 for x in X])
    ax.set_xlabel("Task")
    ax.set_xticklabels([task_name for task_name in task_names])
    #plt.title(task_names[0], fontdict=MONOSPACE_FONT_DICT)

    #plt.xticks([x + len_bars_for_task/2 - cfg['bar_width']/2  - cfg["bar_spacing"]/2 for x in X])
    #plt.xticklabels(task_names, fontdict=MONOSPACE_FONT_DICT)

if data_type == "plan_success":
    ax.set_ylabel("Plan execution success")
else:
    ax.set_ylabel("Plan found")
ax.set_ylim([0,1])
plt.legend()
plt.subplots_adjust(left=0.4)
plt.show()


