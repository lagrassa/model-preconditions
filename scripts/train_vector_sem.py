from isaacgym import gymapi
import matplotlib.pyplot as plt

from plan_abstractions.learning.data_utils import make_vector_datas
from plan_abstractions.models.water_models import *
from autolab_core import YamlConfig
import os


cfg = YamlConfig("cfg/train/train_linear_water_vector.cfg")
processed_datas_train = make_vector_datas(cfg, skill_name=cfg["skill_name"], tag_name="tags")
processed_datas_val = make_vector_datas(cfg, skill_name=cfg["skill_name"], tag_name="val_tags")
init_state_data_train, param_data_train, end_state_data_train = processed_datas_train["init_states"], processed_datas_train["params"], processed_datas_train["end_states"]
init_state_data_val, param_data_val, end_state_data_val = processed_datas_val["init_states"], processed_datas_val["params"], processed_datas_val["end_states"]
#init_state_data_val = np.load("/home/lagrassa/plan_abstractions/good_control_init_states.npy")
#end_state_data_val = np.load("/home/lagrassa/plan_abstractions/good_control_end_states.npy")
#param_data_val = np.load("/home/lagrassa/plan_abstractions/good_control_params.npy")

#init_state_data_train = init_state_data_train[:, :2]
#end_state_data_train = end_state_data_train[:, :2]
model = eval(cfg["model_type"])(model_cfg=cfg["model_cfg"])
model.train(init_state_data_train, param_data_train, end_state_data_train)

pred_end_state_data_val = np.vstack([model.predict(init_state_data_val[i, :].reshape(1, -1), param_data_val[i, :].reshape(1, -1))['end_states'][0] for i in range(len(param_data_val))])
pred_end_state_data_train = np.vstack([model.predict(init_state_data_train[i, :].reshape(1, -1), param_data_train[i, :].reshape(1, -1))['end_states'][0] for i in range(len(param_data_train))])
error_vec = pred_end_state_data_val - end_state_data_val
error_vec_train = pred_end_state_data_train - end_state_data_train
import ipdb; ipdb.set_trace()
error = np.linalg.norm(pred_end_state_data_val - end_state_data_val, axis=1)
error_train = np.linalg.norm(pred_end_state_data_train - end_state_data_train, axis=1)
print("Error stats")
print("Mean error val ", np.mean(error))
print("std error val", np.std(error))

print("Mean error train ", np.mean(error_train))
print("std error train", np.std(error_train))
#plt.hist(error, range=(0,0.2) ,bins=15)
#plt.show()
#plt.scatter(pred_end_state_data_val[:,-2], end_state_data_val[:,-2])
#plt.scatter(pred_end_state_data_val[:,-1], end_state_data_val[:,-1])

f, axes = plt.subplots(nrows=1,ncols=4, sharey=True)
idxs = [0,1,-1,-2]
labels = ["x", "y", "control", "target"]
for label, idx in zip(labels, idxs):
    axes[idx].scatter(end_state_data_val[:,idx], pred_end_state_data_val[:,idx])
    axes[idx].set_xlabel(label)
f.tight_layout()
#plt.scatter(error, error)
plt.show()

