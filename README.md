
### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
    - `conda create -n your_virtual_env python=3.8`
    - `conda activate your_virtual_env`
    - `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
2. Install pytorch 1.10 with cuda-12.1:
    - `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
3. Install Isaac Gym
   - (Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym)
   - `cd isaacgym/isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`) 
4. Install legged_gym
    - Clone this repository
   - `cd ~/TRON1-RL-ISAACGYM-WALK && pip install -e .`
5. Install tensorboard
   - `pip install tensorboard`

### CODE STRUCTURE ###
1. Each environment is defined by an env file `pointfoot_flat.py` and a config file `pointfoot_flat_config.py`(take pointfoot for example). The config file contains two classes: one conatianing all the environment parameters (`BipedCfgPF`) and one for the training parameters (`BipedCfgPPOPF`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
0. `cd TRON1-RL-ISAACGYM-walk && source setup_env.sh` 设置环境变量

1. Train(limx_arm):

    ````
    python legged_gym/scripts/train.py --task=limx_arm --headless
    ```
    ```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `pointfoot-legged-gym/logs/limx_arm/SF_TRON1A_ARM/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
  ```python legged_gym/scripts/play.py --task=pointfoot_flat --load_run your_model_path --checkpoint your_checkpoint```
    - `load_run` is the folder name which contains your training results, for example `Apr18_15-48-46_`
    - `checkpoint` is the number of training iteration, for example the checkpoint of `model_10000.pt` is 10000.


### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```

## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.

## Any Questions?

If you have any more questions, please create an issue in this repository.





# Notes #
Oct24_17-05-56_
减小torque相关的限制
torques = -0.00004 # -0.00008
power = -1e-4 # -2e-4
soft_torque_limit = 0.9 # 0.8
---走得更稳了，但sim2sim腿还是会叉开，机身左右晃动比较大


增加延迟和腿分开惩罚
本体感知延迟 - 角速度、重力、关节位置/速度都有10-30ms的随机延迟
IMU偏移 - ±1.2度的随机IMU安装偏移
动作执行延迟 - 0-20ms的动作执行延迟（原有）
双脚距离约束 - 既惩罚过近（<0.2m）也惩罚过远（>0.5m）_reward_feet_distance(self)
从头开始训
