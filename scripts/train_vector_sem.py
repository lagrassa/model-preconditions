from isaacgym import gymapi
import matplotlib.pyplot as plt
from plan_abstractions.models.water_models import *
from autolab_core import YamlConfig
import os


cfg = YamlConfig("cfg/train/train_linear_water_vector.cfg")
model = eval(cfg['model_type'])(cfg["model_cfg"])

exp_name = "morexvar"
data_root = "data/"
data_dir = os.path.join(data_root, exp_name)
init_state_data = np.load(os.path.join(data_dir, "init_states.npy"))
end_state_data = np.load(os.path.join(data_dir, "end_states.npy"))
param_data = np.load(os.path.join(data_dir, "param_data.npy"))
mask = (param_data < 1.3).flatten()
init_state_data, end_state_data, param_data = init_state_data[mask], end_state_data[mask], param_data[mask]

model.train(init_state_data, param_data, end_state_data)

pred_end_state_data = np.vstack([model.predict(init_state_data[i,:].reshape(1,-1), param_data[i,:].reshape(1,-1)) for i in range(len(param_data))])
error = np.linalg.norm(pred_end_state_data- end_state_data, axis=1)
plt.scatter(error, error)
plt.show()

