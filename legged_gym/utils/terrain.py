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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        if self.type == "trimesh":
            self.tot_cols = (
                int(cfg.num_cols * self.width_per_env_pixels) + 4 * self.border
            )
        else:
            self.tot_cols = (
                int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
            )
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.terrain_num = np.zeros(7, dtype=np.int16)
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            (
                self.vertices,
                self.triangles,
            ) = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )

        # Handle competition terrain type
        if self.type in ['competition']:
            self.border = int(cfg.border_size/self.cfg.horizontal_scale)
            self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) 
            self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) 
            # Initialize terrain_num for competition terrain if needed
            if not hasattr(self, 'terrain_num'):
                self.terrain_num = np.zeros(7, dtype=np.int16)

        
        self.heightsamples = self.height_field_raw
        if self.type in ['trimesh','competition']:
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)

        

    def randomized_terrain(self):
        # Ensure terrain_num is initialized for randomized terrain
        if not hasattr(self, 'terrain_num'):
            self.terrain_num = np.zeros(7, dtype=np.int16)
            
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        # Ensure terrain_num is initialized for curriculum terrain
        if not hasattr(self, 'terrain_num'):
            self.terrain_num = np.zeros(7, dtype=np.int16)
            
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.5
        random_height = 0.05 + difficulty * 0.15
        default_step_width = 0.37  # 28cm
        max_step_height = 0.3
        step_height = 0.05 + difficulty * 0.23
        step_slope = step_height / default_step_width
        discrete_obstacles_height = 0.05 + difficulty * 0.25
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        
        # Ensure terrain_num is initialized
        if not hasattr(self, 'terrain_num'):
            self.terrain_num = np.zeros(7, dtype=np.int16)
            
        if choice < self.proportions[0]:
            self.terrain_num[0] += 1
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[1]:
            self.terrain_num[1] += 1
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,
                max_height=random_height,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            step_scale = np.array([1, 1.05, 0.95, 1.1, 0.9, 1.2, 0.8])
            if choice < self.proportions[2]:
                self.terrain_num[2] += 1
                step_height *= -1
                step_slope *= -1
                scale_idx = min(int((self.terrain_num[2] - 1) / self.cfg.num_rows), len(step_scale) - 1)
                step_width = default_step_width * step_scale[scale_idx]
            else:
                self.terrain_num[3] += 1
                scale_idx = min(int((self.terrain_num[3] - 1) / self.cfg.num_rows), len(step_scale) - 1)
                step_width = default_step_width * step_scale[scale_idx]
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=step_width,
                step_height=np.clip(
                    step_slope * step_width, -max_step_height, max_step_height
                ),
                platform_size=3.0,
            )
        elif choice < self.proportions[4]:
            self.terrain_num[4] += 1
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            self.terrain_num[5] += 1
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[6]:
            self.terrain_num[6] += 1
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0
    
    return terrain

def continuous_gap_terrain(terrain, gap_size, num_gaps=3, gap_spacing=2.0):
    """
    生成水平连续gap，只有深坑没有中心平台
    
    Parameters:
        terrain: 地形对象
        gap_size: 每个gap的大小
        num_gaps: gap数量
        gap_spacing: gap之间的间距
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    gap_spacing = int(gap_spacing / terrain.horizontal_scale)
    
    center_y = terrain.width // 2
    
    # 计算每个gap的边界（只有深坑，没有中心平台）
    gap_half_width = gap_size // 2
    gap_half_length = gap_size // 2
    
    # 计算总宽度和起始位置
    total_gap_width = gap_size  # 每个gap的宽度
    total_width = num_gaps * total_gap_width + (num_gaps - 1) * gap_spacing
    start_x = (terrain.length - total_width) // 2
    
    for i in range(num_gaps):
        # 计算当前gap的起始位置
        gap_start_x = start_x + i * (total_gap_width + gap_spacing)
        gap_end_x = gap_start_x + total_gap_width
        
        # 创建深坑区域（矩形的深坑）
        terrain.height_field_raw[
            gap_start_x:gap_end_x, 
            center_y - gap_half_length:center_y + gap_half_length
        ] = -1000
    
    return terrain


def bridge_gap_terrain(terrain, gap_size, num_gaps=3, bridge_width=0.5):
    """
    生成吊桥式连续gap，gap之间有狭窄的连接桥
    
    Parameters:
        terrain: 地形对象
        gap_size: 每个gap的大小
        num_gaps: gap数量
        bridge_width: 连接桥的宽度
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    bridge_width = int(bridge_width / terrain.horizontal_scale)
    
    # 计算总宽度和起始位置
    total_width = num_gaps * gap_size + (num_gaps - 1) * bridge_width
    start_x = (terrain.length - total_width) // 2
    
    # 先设置整个区域为地面
    terrain.height_field_raw[:, :] = 0
    
    for i in range(num_gaps):
        # 计算当前gap的横向位置
        gap_start_x = start_x + i * (gap_size + bridge_width)
        gap_end_x = gap_start_x + gap_size
        
        # 创建横贯整个宽度的深坑（垂直方向的沟）
        terrain.height_field_raw[
            gap_start_x:gap_end_x,  # 横向范围
            0:terrain.width         # 从最前到最后（横贯整个深度）
        ] = -1000
        
        # 在gap之间创建连接桥（保持为地面，高度为0）
        if i < num_gaps - 1:
            bridge_start_x = gap_end_x
            bridge_end_x = bridge_start_x + bridge_width
            terrain.height_field_raw[
                bridge_start_x:bridge_end_x,  # 桥的横向范围
                0:terrain.width               # 横贯整个深度
            ] = 0
    
    return terrain


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    
    return terrain
