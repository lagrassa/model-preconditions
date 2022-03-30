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
#init_state_data_train = init_state_data_train[:, :2]
#end_state_data_train = end_state_data_train[:, :2]
model = eval(cfg["model_type"])(model_cfg=cfg["model_cfg"])
model.train(init_state_data_train, param_data_train, end_state_data_train)

pred_end_state_data_val = np.vstack([model.predict(init_state_data_val[i, :].reshape(1, -1), param_data_val[i, :].reshape(1, -1)) for i in range(len(param_data_val))])
error = np.linalg.norm(pred_end_state_data_val - end_state_data_val, axis=1)
print("Error stats")
print("Mean error", np.mean(error))
print("std error", np.std(error))
#plt.hist(error, range=(0,0.2) ,bins=15)
#plt.show()
#plt.scatter(pred_end_state_data_val[:,-2], end_state_data_val[:,-2])
#plt.scatter(pred_end_state_data_val[:,-1], end_state_data_val[:,-1])
plt.scatter(param_data_val[:,0], end_state_data_val[:,-2])
import ipdb; ipdb.set_trace()
#plt.scatter(error, error)
plt.show()

