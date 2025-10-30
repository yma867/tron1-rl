# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.terrain import *
from isaacgym.terrain_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float, CubicSpline
)
from .limx_arm_config import BipedCfgSF

import math
from time import time
from warnings import WarningMessage
import numpy as np
import os
from typing import Tuple, Dict
import random


class BipedSF(BaseTask):
    def __init__(
        self, cfg: BipedCfgSF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_acc = (self.base_lin_vel - self.last_base_lin_vel) / self.dt
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, self.base_lin_acc)

        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        # self.dof_jerk = (self.last_dof_acc - self.dof_acc) / self.dt
        
        # Update sensor delay FIFOs
        if self.sensor_delay_idx is not None:
            self.ang_vel_fifo = torch.cat(
                (self.base_ang_vel.unsqueeze(1), self.ang_vel_fifo[:, :-1, :]), dim=1
            )
            self.projected_gravity_fifo = torch.cat(
                (self.projected_gravity.unsqueeze(1), self.projected_gravity_fifo[:, :-1, :]), dim=1
            )
            self.dof_pos_fifo = torch.cat(
                (self.dof_pos.unsqueeze(1), self.dof_pos_fifo[:, :-1, :]), dim=1
            )
            self.dof_vel_fifo = torch.cat(
                (self.dof_vel.unsqueeze(1), self.dof_vel_fifo[:, :-1, :]), dim=1
            )

        self.compute_foot_state()
        
        # Update bridge checkpoint detection
        self._update_bridge_checkpoints()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        self._post_physics_step_callback()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # self.last_dof_acc[:] = self.dof_acc[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]

    def compute_foot_state(self):
        super().compute_foot_state()
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf, self.critic_obs_buf = self.compute_self_observations()

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1
        )

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:14] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[14:22] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[22:] = 0.0  # previous actions
        return noise_vec

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        contact_names = []
        if hasattr(self.cfg.asset, "contact_name"):
            contact_names = [s for s in body_names if self.cfg.asset.contact_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.whole_body_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.contact_indices = torch.zeros(
            len(contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )
        for i in range(len(contact_names)):
            self.contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], contact_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

        elif self.cfg.terrain.mesh_type in ["gap"]:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # Calculate terrain dimensions
            terrain_width = self.cfg.terrain.terrain_width
            terrain_length = self.cfg.terrain.terrain_length
            horizontal_scale = self.cfg.terrain.horizontal_scale
            num_rows = int(terrain_width/horizontal_scale)
            
            # Place robots at the end of the first terrain section, just before gap terrain
            # X range: last 20% of the first terrain section (before gap terrain)
            # Y range: middle 80% of terrain length
            x_start = num_rows * 0.1   # 从10%位置开始
            x_end = num_rows * 0.3    # 到30%位置结束
            y_start = int(terrain_length/horizontal_scale) * 0.1  # 从10%位置开始
            y_end = int(terrain_length/horizontal_scale) * 0.9    # 到90%位置结束
            
            self.env_origins[:,0:1] = torch_rand_float(x_start, x_end, (self.num_envs,1), device=self.device) * horizontal_scale
            self.env_origins[:,1:2] = torch_rand_float(y_start, y_end, (self.num_envs,1), device=self.device) * horizontal_scale
            
            # Set Z position slightly above ground
            self.env_origins[:,2:3] = 0.1

        elif self.cfg.terrain.mesh_type in ["competition"]:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # Calculate terrain dimensions
            terrain_width = self.cfg.terrain.terrain_width
            terrain_length = self.cfg.terrain.terrain_length
            horizontal_scale = self.cfg.terrain.horizontal_scale
            num_rows = int(terrain_width/horizontal_scale)
            
            # Place robots at the end of the first terrain section, just before gap terrain
            # X range: last 20% of the first terrain section (before gap terrain)
            # Y range: middle 80% of terrain length
            x_start = num_rows * 0.1   # 从80%位置开始
            x_end = num_rows * 0.3    # 到95%位置结束
            y_start = int(terrain_length/horizontal_scale) * 0.1  # 从10%位置开始
            y_end = int(terrain_length/horizontal_scale) * 0.9    # 到90%位置结束
            
            self.env_origins[:,0:1] = torch_rand_float(x_start, x_end, (self.num_envs,1), device=self.device) * horizontal_scale
            self.env_origins[:,1:2] = torch_rand_float(y_start, y_end, (self.num_envs,1), device=self.device) * horizontal_scale
            
            # Set Z position slightly above ground
            self.env_origins[:,2:3] = 0.1

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._check_walk_stability(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_self_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # Reset progressive checkpoint tracking
        self.last_x_position[env_ids] = self.base_position[env_ids, 0]
        self.bridge_quarter_reached[env_ids] = False
        self.bridge_middle_reached[env_ids] = False
        self.bridge_end_reached[env_ids] = False
        
        # Reset checkpoint reward flags
        self.bridge_quarter_reward_given[env_ids] = False
        self.bridge_middle_reward_given[env_ids] = False
        self.bridge_end_reward_given[env_ids] = False
        
        # Update gap curriculum based on success rate
        if self.cfg.terrain.gap_curriculum and len(env_ids) > 0:
            self._update_gap_curriculum(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        n = env_ids.size(0)
        indices = torch.randperm(n)

        half_size = n // 2
        half_indices = indices[:half_size]
        remaining_indices = indices[half_size:]

        half_list = env_ids[half_indices]
        remaining_list = env_ids[remaining_indices]
        self.dof_pos[half_list] = self.default_dof_pos[half_list, :] + torch_rand_float(
            -0.5, 0.5, (len(half_list), self.num_dof), device=self.device
        )
        self.dof_pos[remaining_list] = self.init_stand_dof_pos[remaining_list, :] + torch_rand_float(
            -0.5, 0.5, (len(remaining_list), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale, # 5 commands
            self.critic_obs_buf
        )

    def compute_self_observations(self):
        # note that observation noise need to modified accordingly !!!
        
        # Use delayed sensor data if sensor delay is enabled
        if self.sensor_delay_idx is not None:
            # Get delayed sensor readings from FIFO buffers
            delayed_ang_vel = self.ang_vel_fifo[torch.arange(self.num_envs), self.sensor_delay_idx, :]
            delayed_projected_gravity = self.projected_gravity_fifo[torch.arange(self.num_envs), self.sensor_delay_idx, :]
            delayed_dof_pos = self.dof_pos_fifo[torch.arange(self.num_envs), self.sensor_delay_idx, :]
            delayed_dof_vel = self.dof_vel_fifo[torch.arange(self.num_envs), self.sensor_delay_idx, :]
        else:
            # Use real-time sensor data (no delay)
            delayed_ang_vel = self.base_ang_vel
            delayed_projected_gravity = self.projected_gravity
            delayed_dof_pos = self.dof_pos
            delayed_dof_vel = self.dof_vel
        
        obs_buf = torch.cat(
            (
                delayed_ang_vel * self.obs_scales.ang_vel,
                delayed_projected_gravity,
                (delayed_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                delayed_dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        # compute critic_obs_buf
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    def get_observations(self):
        return (
            self.obs_buf,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale, # 5 commands
            self.critic_obs_buf
        )


    def create_gap_map(self):

        num_terains = 1
        terrain_width = self.cfg.terrain.terrain_width
        terrain_length = self.cfg.terrain.terrain_length
        horizontal_scale = self.cfg.terrain.horizontal_scale
        vertical_scale = self.cfg.terrain.vertical_scale
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        # Initialize heightsamples with correct dimensions
        total_rows = num_terains * num_rows
        self.terrain.heightsamples = np.zeros((total_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        # Use current curriculum level for number of gaps
        if hasattr(self, 'current_num_gaps'):
            num_gaps = self.current_num_gaps
        else:
            num_gaps = self.cfg.terrain.initial_num_gaps
        
        self.terrain.heightsamples[0:num_rows,:] = bridge_gap_terrain(new_sub_terrain(), gap_size=0.05, num_gaps=num_gaps, bridge_width=0.25).height_field_raw



        self.terrain.vertices, self.terrain.triangles = convert_heightfield_to_trimesh(self.terrain.heightsamples, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -0.
        tm_params.transform.p.y = -0.
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(), self.terrain.triangles.flatten(), tm_params)

        self.width_per_env_pixels = int(terrain_width / horizontal_scale)
        self.length_per_env_pixels = int(terrain_length / horizontal_scale)

        self.border = int(self.cfg.terrain.border_size/self.cfg.terrain.horizontal_scale)
        self.tot_cols = int(self.cfg.terrain.num_cols * self.width_per_env_pixels) #+ 2 * self.border
        self.tot_rows = int(self.cfg.terrain.num_rows * self.length_per_env_pixels) #+ 2 * self.border

        # Use the heightsamples directly as height_field_raw for competition terrain
        self.height_field_raw = self.terrain.heightsamples
        self.tot_rows, self.tot_cols = self.height_field_raw.shape

        self.height_samples = torch.tensor(self.height_field_raw).view(self.tot_rows, self.tot_cols).to(self.device)
    def create_competition_map(self):
        num_terains = 8 #对应下方创建地形的数量
        terrain_width = self.cfg.terrain.terrain_width
        terrain_length = self.cfg.terrain.terrain_length
        horizontal_scale = self.cfg.terrain.horizontal_scale
        vertical_scale = self.cfg.terrain.vertical_scale
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        # Initialize heightsamples with correct dimensions
        total_rows = num_terains * num_rows
        self.terrain.heightsamples = np.zeros((total_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        self.terrain.heightsamples[0:num_rows, :] =  sloped_terrain(new_sub_terrain(), slope=0.0).height_field_raw
        self.terrain.heightsamples[num_rows:2*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.3).height_field_raw
        #self.terrain.heightsamples[num_rows:2*num_rows, :] = sloped_terrain(new_sub_terrain(), slope=0.1).height_field_raw
        self.terrain.heightsamples[2*num_rows:3*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.15, max_height=0.15, step=0.2, downsampled_scale=0.5).height_field_raw
        self.terrain.heightsamples[3*num_rows:4*num_rows,:] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.15, min_size=1., max_size=5., num_rects=20).height_field_raw
        self.terrain.heightsamples[4*num_rows:5*num_rows,:] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw
        self.terrain.heightsamples[5*num_rows:6*num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=0.25).height_field_raw
        self.terrain.heightsamples[6*num_rows:7*num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.25).height_field_raw
        #self.terrain.heightsamples[6*num_rows:7*num_rows,:] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
        self.terrain.heightsamples[7*num_rows:8*num_rows,:] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
                                                                        stone_distance=0.25, max_height=0.2, platform_size=0.).height_field_raw

        
        self.terrain.vertices, self.terrain.triangles = convert_heightfield_to_trimesh(self.terrain.heightsamples, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -0.
        tm_params.transform.p.y = -0.
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(), self.terrain.triangles.flatten(), tm_params)

        self.width_per_env_pixels = int(terrain_width / horizontal_scale)
        self.length_per_env_pixels = int(terrain_length / horizontal_scale)

        self.border = int(self.cfg.terrain.border_size/self.cfg.terrain.horizontal_scale)
        self.tot_cols = int(self.cfg.terrain.num_cols * self.width_per_env_pixels) #+ 2 * self.border
        self.tot_rows = int(self.cfg.terrain.num_rows * self.length_per_env_pixels) #+ 2 * self.border

        # Use the heightsamples directly as height_field_raw for competition terrain
        self.height_field_raw = self.terrain.heightsamples
        self.tot_rows, self.tot_cols = self.height_field_raw.shape

        self.height_samples = torch.tensor(self.height_field_raw).view(self.tot_rows, self.tot_cols).to(self.device)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh','competition','gap']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type=='competition':
            self.create_competition_map()
        elif mesh_type=='gap':
            self.create_gap_map()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh, competition, gap]")
        self._create_envs()

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        if self.cfg.terrain.mesh_type in ['competition', 'gap']:
            points = (points/self.terrain.cfg.horizontal_scale).long()
        else:
            points += self.terrain.cfg.border_size
            points = (points/self.terrain.cfg.horizontal_scale).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        #pos = x,y
        # ,self.height_samples[x/0.25,y/0.25]
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                    self.episode_length_buf
                    % int(self.cfg.commands.resampling_time / self.dt)
                    == 0
            )
                .nonzero(as_tuple=False)
                .flatten()
        )
        self._resample_commands(env_ids, False)
        self._resample_gaits(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 1.0 * wrap_to_pi(self.commands[:, 5] - heading)

        self._resample_zero_commands(env_ids)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _step_contact_targets(self):
        super()._step_contact_targets()
        self._generate_des_ee_ref()

    def _generate_des_ee_ref(self):
        frequencies = self.gaits[:, 0]
        mask_0 = (self.gait_indices < 0.25) & (self.gait_indices >= 0.0)  # lift up
        mask_1 = (self.gait_indices < 0.5) & (self.gait_indices >= 0.25)  # touch down
        mask_2 = (self.gait_indices < 0.75) & (self.gait_indices >= 0.5)  # lift up
        mask_3 = (self.gait_indices <= 1.0) & (self.gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.cfg.gait.touch_down_vel
        swing_end_vel[mask_3] = self.cfg.gait.touch_down_vel

        # generate desire foot z trajectory
        swing_height = self.gaits[:, 3]
        # self.des_foot_height = 0.5 * swing_height * (1 - torch.cos(4 * np.pi * self.gait_indices))
        # self.des_foot_velocity_z = 2 * np.pi * swing_height * frequencies * torch.sin(
        #     4 * np.pi * self.gait_indices)

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(self.gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(self.gait_indices / frequencies)

    def _resample_gaits(self, env_ids):
        super()._resample_gaits(env_ids)
        self._resample_stand_still_gait_commands(env_ids)

    def _check_walk_stability(self, env_ids):
        if len(env_ids) != 0:
            self.mean_episode_len = torch.mean(self.episode_length_buf[env_ids].float(), dim=0).cpu().item()
        if self.mean_episode_len > 950:
            self.stable_episode_length_count += 1
            # print("Stable Episode Length:{}, count:{}.".format(self.mean_episode_len, self.stable_episode_length_count))
        else:
            self.stable_episode_length_count = 0

    def _update_bridge_checkpoints(self):
        """Update bridge checkpoint detection based on robot position"""
        current_x = self.base_position[:, 0]
        
        # Calculate terrain dimensions
        terrain_width = self.cfg.terrain.terrain_width
        horizontal_scale = self.cfg.terrain.horizontal_scale
        num_rows = int(terrain_width / horizontal_scale)
        
        # Bridge terrain spans the entire terrain (since num_terains = 1)
        bridge_start = 0.0
        bridge_length = num_rows * horizontal_scale
        
        # Define checkpoint positions
        quarter_pos = bridge_start + bridge_length * 0.25
        middle_pos = bridge_start + bridge_length * 0.5
        end_pos = bridge_start + bridge_length * 0.8  # 80% through is considered "end"
        
        # Check if robots have reached each checkpoint (progressive)
        reached_quarter = (current_x >= quarter_pos) & (self.last_x_position < quarter_pos)
        reached_middle = (current_x >= middle_pos) & (self.last_x_position < middle_pos)
        reached_end = (current_x >= end_pos) & (self.last_x_position < end_pos)
        
        # Update checkpoint status (once reached, stays true until reset)
        self.bridge_quarter_reached = self.bridge_quarter_reached | reached_quarter
        self.bridge_middle_reached = self.bridge_middle_reached | reached_middle
        self.bridge_end_reached = self.bridge_end_reached | reached_end
        
        # Update last position for next frame
        self.last_x_position = current_x.clone()

    def _update_gap_curriculum(self, env_ids):
        """Update gap curriculum based on success rate"""
        # Count successful episodes (those that reached the bridge end)
        successful_envs = self.bridge_end_reached[env_ids]
        success_count = torch.sum(successful_envs).item()
        
        # Update statistics
        self.episode_success_count += success_count
        self.total_episode_count += len(env_ids)
        
        # Check if we have enough episodes to evaluate
        if self.total_episode_count >= self.cfg.terrain.min_episodes_for_evaluation:
            current_success_rate = self.episode_success_count / self.total_episode_count
            self.success_rate_history.append(current_success_rate)
            
            # Check if we should advance to next curriculum level
            if current_success_rate >= self.cfg.terrain.success_rate_threshold:
                if self.current_num_gaps == self.cfg.terrain.initial_num_gaps:
                    # Advance from 1 gap to 3 gaps
                    self.current_num_gaps = self.cfg.terrain.intermediate_num_gaps
                    #print(f"Gap Curriculum: Advanced to {self.current_num_gaps} gaps! Success rate: {current_success_rate:.3f}")
                    self._reset_curriculum_stats()
                    self._regenerate_terrain()
                    
                elif self.current_num_gaps == self.cfg.terrain.intermediate_num_gaps:
                    # Advance from 3 gaps to 5 gaps
                    self.current_num_gaps = self.cfg.terrain.final_num_gaps
                    #print(f"Gap Curriculum: Advanced to {self.current_num_gaps} gaps! Success rate: {current_success_rate:.3f}")
                    self._reset_curriculum_stats()
                    self._regenerate_terrain()
                    
                # If already at final level, just reset stats
                elif self.current_num_gaps == self.cfg.terrain.final_num_gaps:
                    #print(f"Gap Curriculum: Maintaining {self.current_num_gaps} gaps. Success rate: {current_success_rate:.3f}")
                    self._reset_curriculum_stats()
            
            # If success rate is too low, reset stats but keep current level
            else:
                #print(f"Gap Curriculum: Current level {self.current_num_gaps} gaps. Success rate: {current_success_rate:.3f} (need {self.cfg.terrain.success_rate_threshold:.1f})")
                self._reset_curriculum_stats()

    def _reset_curriculum_stats(self):
        """Reset curriculum statistics for next evaluation period"""
        self.episode_success_count = 0
        self.total_episode_count = 0

    def _regenerate_terrain(self):
        """Regenerate terrain with new gap configuration"""
        # Clear existing terrain
        self.gym.clear_lines(self.viewer) if hasattr(self, 'viewer') and self.viewer else None
        
        # Recreate terrain with new gap count
        self.create_gap_map()
        
        print(f"Terrain regenerated with {self.current_num_gaps} gaps")

    def _resample_commands(self, env_ids, is_start=True):
        """Randommly select commands of some environments

                Args:
                    env_ids (List[int]): Environments ids for which new commands are needed
                """
        self.commands[env_ids, 0] = (self.command_ranges["lin_vel_x"][env_ids, 1]
                                     - self.command_ranges["lin_vel_x"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_x"][env_ids, 0]
        self.commands[env_ids, 1] = (self.command_ranges["lin_vel_y"][env_ids, 1]
                                     - self.command_ranges["lin_vel_y"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_y"][env_ids, 0]
        self.commands[env_ids, 2] = (self.command_ranges["ang_vel_yaw"][env_ids, 1]
                                     - self.command_ranges["ang_vel_yaw"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["ang_vel_yaw"][env_ids, 0]
        self.commands[env_ids, 3] = (self.command_ranges["base_height"][env_ids, 1]
                                     - self.command_ranges["base_height"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["base_height"][env_ids, 0]

        self._resample_stand_still_commands(env_ids, is_start)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 5] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _resample_zero_commands(self, env_ids):
        thresh = 0.25
        indices_to_update = env_ids[(self.commands[env_ids, 0] < thresh) & (self.commands[env_ids, 0] > -thresh)]
        self.commands[indices_to_update, :3] = 0.0

    def _resample_stand_still_commands(self, env_ids, is_start=True):
        if (not self.walk_stability) and self.stable_episode_length_count >= 10:
            self.walk_stability = True
            self.stable_episode_length_count = 0
        if self.walk_stability and (not self.stand_still_stability) and self.stable_episode_length_count >= 10:
            self.stand_still_stability = True
        if self.walk_stability and (not self.stand_still_stability):
            if not is_start:
                indices_to_update = env_ids[self.commands[env_ids, 4] == 0]
                self.commands[indices_to_update, 4] = (self.command_ranges["stand_still"][indices_to_update, 1]
                                                       - self.command_ranges["stand_still"][indices_to_update, 0]) \
                                                      * torch.randint(0, 2, (len(indices_to_update),),
                                                                      device=self.device) \
                                                      + self.command_ranges["stand_still"][indices_to_update, 0]
                indices_to_update1 = indices_to_update[self.commands[indices_to_update, 4] == 1]
                self.commands[indices_to_update1, :3] = 0.0
            else:
                self.commands[env_ids, 4] = 0
        elif self.walk_stability and self.stand_still_stability:
            self.commands[env_ids, 4] = (self.command_ranges["stand_still"][env_ids, 1]
                                         - self.command_ranges["stand_still"][env_ids, 0]) \
                                        * torch.randint(0, 2, (len(env_ids),), device=self.device) \
                                        + self.command_ranges["stand_still"][env_ids, 0]
            indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
            self.commands[indices_to_update, :3] = 0.0
        else:
            self.commands[env_ids, 4] = 0

    def _resample_stand_still_gait_commands(self, env_ids):
        # indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
        # self.gaits[indices_to_update, :] = 0.0
        pass

    def _resample_stand_still_gait_clock(self):
        indices_to_update = torch.nonzero(self.commands[:, 4] == 1).squeeze()
        gait_indices = self.gait_indices[indices_to_update]

        mask_0_5_to_0_55 = (gait_indices >= 0.5) & (gait_indices < 0.55)
        mask_0_0_to_0_05 = (gait_indices >= 0.0) & (gait_indices < 0.05)
        mask_0_95_to_1_0 = (gait_indices >= 0.95) & (gait_indices < 1.0)

        self.gait_indices[indices_to_update[mask_0_5_to_0_55]] = 0.5
        self.gait_indices[indices_to_update[mask_0_0_to_0_05]] = 0.0
        self.gait_indices[indices_to_update[mask_0_95_to_1_0]] = 0.0

        mask_else = ~(mask_0_5_to_0_55 | mask_0_0_to_0_05 | mask_0_95_to_1_0)
        self.commands[indices_to_update[mask_else], 4] = 0

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_heights = torch.zeros_like(self.foot_positions[:, :, 2])
        self.last_base_lin_vel = self.base_lin_vel.clone()

        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.variances_per_env = 0
        self.init_stand_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            if hasattr(self.cfg.init_state, "init_stand_joint_angles"):
                stand_angle = self.cfg.init_state.init_stand_joint_angles[name]
                self.init_stand_dof_pos[:, i] = stand_angle

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1, 1],
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"][:] = torch.tensor(
            self.cfg.commands.ranges.base_height
        )
        self.command_ranges["stand_still"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["stand_still"][:] = torch.tensor(
            self.cfg.commands.ranges.stand_still
        )

        self.des_foot_height = torch.zeros(self.num_envs,
                                           dtype=torch.float,
                                           device=self.device, requires_grad=False, ) # TODO
        self.des_foot_velocity_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                               requires_grad=False, ) # TODO
        
        # Progressive checkpoint tracking buffers
        self.last_x_position = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.bridge_quarter_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.bridge_middle_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.bridge_end_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # Reward given flags for each checkpoint
        self.bridge_quarter_reward_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.bridge_middle_reward_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.bridge_end_reward_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # Gap curriculum tracking
        if self.cfg.terrain.gap_curriculum:
            self.current_num_gaps = self.cfg.terrain.initial_num_gaps
            self.episode_success_count = 0
            self.total_episode_count = 0
            self.success_rate_history = []

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        self.rwd_orientationPrev = self._reward_orientation()
        # self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_baseHeightPrev = self._reward_base_height()
        if "tracking_contacts_shaped_height" in self.reward_scales.keys():
            self.rwd_swingHeightPrev = self._reward_tracking_contacts_shaped_height()

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)
    
    # ----------------------rewards----------------------
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                    )
                )

        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                    )
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (1 - torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma)
                )
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_height(self):
        foot_heights = self.foot_heights
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_height"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma)
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma)
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma))
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)



    # def _reward_feet_distance(self):
    #     # Penalize base height away from target
    #     feet_distance = torch.norm(
    #         self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
    #     )
    #     return torch.where(
    #         torch.logical_and(self.commands[:, 4] == 1, self.commands[:, 3] <= 0.3),
    #         torch.clip(torch.abs(self.cfg.rewards.min_feet_distance - feet_distance), 0, 1),
    #         torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1),
    #     )
    # 低姿态时惩罚双脚距离过近，鼓励叉开保持平衡

    def _reward_feet_distance(self):
        # Penalize feet distance away from target range [min_feet_distance, max_feet_distance]
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        
        # Penalty for feet too close (below min_feet_distance)
        too_close_penalty = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        
        # Penalty for feet too far (above max_feet_distance)
        too_far_penalty = torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        
        # Combined penalty: penalize both too close and too far
        penalty = too_close_penalty + too_far_penalty
        
        return torch.where(
            torch.logical_and(self.commands[:, 4] == 1, self.commands[:, 3] <= 0.3),
            torch.clip(torch.abs(self.cfg.rewards.min_feet_distance - feet_distance), 0, 1),
            penalty,
        )

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.025
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward

    def _reward_power(self):
        # Penalize torques
        joint_array = [i for i in range(self.num_dof)]
        joint_array.remove(3)
        joint_array.remove(7)
        return torch.sum(torch.abs(self.torques[:, joint_array] * self.dof_vel[:, joint_array]), dim=1)

    def _reward_collision(self):
        reward = torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        # reward = torch.square(base_height - self.commands[:, 3])
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)
        # return torch.where(self.commands[:, 4] == 0, reward, reward * 1.5)
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_ankle_torque_limits(self):
        torque_limit = torch.cat((self.torque_limits[3].view(1) * self.cfg.rewards.soft_torque_limit,
                                  self.torque_limits[7].view(1) * self.cfg.rewards.soft_torque_limit),
                                 dim=-1, )
        torque = torch.cat((self.torques[:, 3].view(self.num_envs, 1),
                            self.torques[:, 7].view(self.num_envs, 1)), dim=-1)
        return torch.sum(
            torch.pow(torque / torque_limit, 8),
            dim=1,
        )

    def _reward_relative_feet_height_tracking(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        feet_height_in_body_frame = base_height.view(self.num_envs, 1) - self.foot_heights
        reward = torch.exp(
            -torch.sum(
                torch.square(
                    feet_height_in_body_frame - self.commands[:, 3].view(self.num_envs, 1)
                ),
                dim=-1) / self.cfg.rewards.height_tracking_sigma
        )
        return torch.where(self.commands[:, 4] == 1, reward, 0)

    def _reward_zero_command_nominal_state(self):
        # Penalize the hip joint pos in zero command
        dof_pos = self.dof_pos - self.raw_default_dof_pos
        reward = torch.sum(
            torch.square(dof_pos[:, [1, 5]]), dim=1
        )
        return reward * torch.logical_and(torch.norm(self.commands[:, :3], dim=1) < 0.05, self.commands[:, 4] == 0)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_keep_ankle_pitch_zero_in_air(self):
        ankle_pitch = torch.abs(self.dof_pos[:, 3]) * ~self.contact_filt[:, 0] + torch.abs(
            self.dof_pos[:, 7]) * ~self.contact_filt[:, 1]
        return torch.exp(-torch.abs(ankle_pitch) / 0.2)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~(self.time_out_buf | self.edge_reset_buf)

    def _reward_fail(self):
        return self.fail_buf > 0

    def _reward_keep_balance(self):
        #给予持续的正奖励来激励agent保持稳定状态，只要存活就会得到这个奖励，以延长存活时间
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
    

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        torque_limit = self.torque_limits * self.cfg.rewards.soft_torque_limit
        return torch.sum(
            torch.pow(self.torques / torque_limit, 8),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_stand_still(self):

        return torch.sum(self.foot_heights, dim=1) * (
            torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.min_norm
        )

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                self.contact_forces[:, self.feet_indices, 2]
                - self.base_mass.mean() * 9.8 / 2
            ).clip(min=0.0),
            dim=1,
        )

    

    def _reward_bridge_quarter(self):
        """Reward for reaching 1/4 of the bridge"""
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Check which robots just reached quarter and haven't received reward yet
        just_reached = self.bridge_quarter_reached & (~self.bridge_quarter_reward_given)
        
        # Give reward to robots that just reached the checkpoint
        reward[just_reached] = 1.0
        
        # Mark that reward has been given
        self.bridge_quarter_reward_given = self.bridge_quarter_reward_given | just_reached
        
        return reward

    def _reward_bridge_middle(self):
        """Reward for reaching middle of the bridge"""
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Check which robots just reached middle and haven't received reward yet
        just_reached = self.bridge_middle_reached & (~self.bridge_middle_reward_given)
        
        # Give reward to robots that just reached the checkpoint
        reward[just_reached] = 1.0
        
        # Mark that reward has been given
        self.bridge_middle_reward_given = self.bridge_middle_reward_given | just_reached
        
        return reward

    def _reward_bridge_end(self):
        """Reward for completing the bridge"""
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Check which robots just reached end and haven't received reward yet
        just_reached = self.bridge_end_reached & (~self.bridge_end_reward_given)
        
        # Give reward to robots that just reached the checkpoint
        reward[just_reached] = 1.0
        
        # Mark that reward has been given
        self.bridge_end_reward_given = self.bridge_end_reward_given | just_reached
        
        return reward

    def _reward_stationary_penalty(self):
        """Penalty for staying stationary (not moving forward)"""
        # Calculate forward velocity (x-direction)
        forward_vel = self.base_lin_vel[:, 0]
        
        # Define minimum forward velocity threshold
        min_forward_vel = 0.5  # m/s
        
        # Penalize when forward velocity is below threshold
        stationary_mask = forward_vel < min_forward_vel
        
        # Calculate penalty based on how far below the threshold
        velocity_deficit = torch.clamp(min_forward_vel - forward_vel, min=0.0)
        
        # Apply penalty only when robot is stationary
        reward = torch.where(
            stationary_mask,
            velocity_deficit,  # Penalty proportional to velocity deficit
            torch.zeros_like(forward_vel)  # No penalty when moving forward
        )
        
        return reward

    def _reward_stand_up_reward(self):
        """Reward for standing up from squatting position"""
        # Calculate the height achievement (normalized by target height)
        height_error = torch.abs(self.base_height - self.cfg.rewards.base_height_target)
        height_reward = torch.exp(-height_error / 0.1)  # Exponential reward for being close to target height
        
        # Reward for achieving target joint positions (standing posture)
        joint_pos_error = torch.sum(torch.square(self.dof_pos - self.raw_default_dof_pos), dim=1)
        joint_pos_reward = torch.exp(-joint_pos_error / 0.5)
        
        # Combined reward (only when both conditions are met)
        reward = height_reward * joint_pos_reward
        
        return reward

    