pytest tests/test_controllers.py
pytest tests/test_franka_env.py
pytest tests/test_franka_skills.py
pytest tests/test_rod_env.py
pytest tests/test_rod_skills.py
pytest tests/test_skill_sem_preds.py
pytest -k test_PD_free_space_move tests/test_franka_controllers.py
pytest -k test_lqr_free_space_move tests/test_franka_controllers.py
pytest -k test_lqr_waypoints_xyz2 tests/test_franka_controllers.py
pytest -k test_lqr_waypoints_xyz_yaw2 tests/test_franka_controllers.py

