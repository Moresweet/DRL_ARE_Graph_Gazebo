import timeit
import time
# from datetime import time

import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from std_msgs.msg import Int8

from gazebo_env import GazeboEnv
from model import PolicyNet
from test_parameter import *

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = GazeboEnv(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=True)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.robot_position = self.env.robot_position
        self.perf_metrics = dict()

        self.center_points_frontier = None
        self.center_frontiers_sub = rospy.Subscriber(
            "/center_frontiers_pixel", Marker, self.get_center_frontiers_callback, queue_size=1
        )
        self.reset_nbv_pub = rospy.Publisher("/reset_nbv", Int8, queue_size=1)
        self.update_center_frontier = True
        self.exploration_time = []

    def select_closest_frontier(self, next_position):
        while self.update_center_frontier is True:
            pass
        closest_point = None
        closest_distance = float("inf")

        for point in self.center_points_frontier:
            distance = np.linalg.norm(np.array([point.x, point.y]) - next_position)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point

        return np.array([closest_point.x, closest_point.y])

    def get_center_frontiers_callback(self, center_frontier_maker):
        if self.update_center_frontier is True:
            self.center_points_frontier = center_frontier_maker.points
            self.update_center_frontier = False

    def run_episode(self, curr_episode):
        done = False
        no_nbv = False
        observations = self.get_observations()
        self.exploration_time = []
        for i in range(30):
            next_position, action_index = self.select_node(observations)

            policy_center_frontier = self.select_closest_frontier(next_position)
            start_time = timeit.default_timer()
            reward, done, self.robot_position, self.travel_dist, plan_status, no_nbv, object_reward = self.env.step(
                policy_center_frontier,
                next_position,
                self.travel_dist)
            end_time = timeit.default_timer()
            time_elapsed = end_time - start_time
            time_elapsed_sec = time_elapsed
            self.exploration_time.append(time_elapsed_sec)
            observations = self.get_observations()

            if no_nbv:
                reset_msg = Int8()
                reset_msg.data = 1
                self.reset_nbv_pub.publish(reset_msg)
                time.sleep(3)
                no_nbv = False
            self.update_center_frontier = True
            # 簇中心点不再出现
            start_time = timeit.default_timer()
            mutex = True
            while self.update_center_frontier is True and mutex is True:
                end_time = timeit.default_timer()
                execution_time = end_time - start_time
                if execution_time > 5:
                    # 重新唤醒nbv
                    # 跳出循环
                    no_nbv = True
                    mutex = False
            # save evaluation data
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = f'results/trajectory/ours_trajectory_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist', 'area', 'time']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist, np.sum(self.env.robot_belief == 255), time_elapsed_sec]).reshape(1, -1)
                    writer.writerows(csv_data)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            # 边界获取不到特殊处理，也属于完成了
            if no_nbv is True and self.env.explored_area > 0.95 and done is False:
                done = True
                print("done")

            if done or no_nbv or self.env.plan_filed_count >= 2:
                break
            reset_msg = Int8()
            reset_msg.data = 1
            self.env.reset_nbv_pub.publish(reset_msg)
            time.sleep(3)

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_area'] = self.env.explored_area
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['object_reward'] = object_reward

        # save final path length
        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = f'results/length/ours_length_result.csv'
            new_file = False if os.path.exists(csv_filename) else True
            field_names = ['dist']
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.travel_dist]).reshape(-1, 1)
                writer.writerows(csv_data)

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)
        self.env.reset(5)
        reset_msg = Int8()
        reset_msg.data = 1
        self.env.reset_nbv_pub.publish(reset_msg)
        time.sleep(3)

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)

        # 添加目标观测输入
        object_value = copy.deepcopy(self.env.object_value)

        # normalize observations
        node_coords = node_coords / 384
        node_utility = node_utility / 120
        # 6类，最大到19
        object_value = object_value / 19

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        node_object_value_inputs = object_value.reshape((n_nodes, 1))
        # 对齐维度
        node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost, node_object_value_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # calculate a mask for padded node
        node_padding_mask = None

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        try:
            # 防止悬空节点的选择导致回原点
            if len(graph[str(current_node_index)]) <= 1:
                distances = np.linalg.norm(self.env.node_coords - self.robot_position, axis=1)
                indices = np.where(distances < np.sort(distances))[0]
                for sub_index in indices:
                    if len(graph[str(sub_index)]) > 1:
                        current_node_index = sub_index
                        break
        except IndexError:
            return None
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)


        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(0)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_area), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

    def work(self, curr_episode):
        self.run_episode(curr_episode)
        self.env.reset(5)
