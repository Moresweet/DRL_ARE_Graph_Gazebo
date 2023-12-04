import copy
import os
import time
import timeit

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt

from gazebo_env import GazeboEnv
import rospy
from std_msgs.msg import Int8
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from parameter import *


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = GazeboEnv(map_index=self.global_step, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.robot_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        self.center_points_frontier = None
        self.center_frontiers_sub = rospy.Subscriber(
            "/center_frontiers_pixel", Marker, self.get_center_frontiers_callback, queue_size=1
        )
        self.reset_nbv_pub = rospy.Publisher("/reset_nbv", Int8, queue_size=1)
        self.update_center_frontier = True
        # 15如何来，是这么来的，在worker中，经验池有15项数据
        for i in range(15):
            self.episode_buffer.append([])

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
        if closest_point is None:
            pass
        return np.array([closest_point.x, closest_point.y])

    def get_center_frontiers_callback(self, center_frontier_maker):
        if self.update_center_frontier is True:
            self.center_points_frontier = center_frontier_maker.points
            self.update_center_frontier = False

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
        node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost, node_object_value_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # padding the number of node to a given node padding size
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        try:
            # 防止悬空节点的选择导致回原点
            if len(graph[current_node_index]) <= 1:
                distances = np.linalg.norm(self.env.node_coords - self.robot_position, axis=1)
                indices = np.where(distances < np.sort(distances))[0]
                for sub_index in indices:
                    if len(graph[sub_index]) > 1:
                        current_node_index = sub_index
                        break
        except IndexError:
            return None
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        # 上次重写while后不会出现这个溢出了，但是怀疑是会出现没有的情况
        if current_index > edge_inputs.__len__():
            print("数组要溢出了")
            # 暂时来看这里也可以作为里程计和地图出错的判断标准
            return None
        try:
            edge = edge_inputs[current_index]
        except IndexError:
            return None
        edge = edge_inputs[current_index]
        if edge is not None:
            while len(edge) < self.k_size:
                edge.append(0)

            edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

            # calculate a mask for the padded edges (denoted by 0)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

            observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
            return observations
        else:
            return None

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

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask)
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask)
        self.episode_buffer[5] += copy.deepcopy(edge_mask)

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[8] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[9] += copy.deepcopy(node_inputs)
        self.episode_buffer[10] += copy.deepcopy(edge_inputs)
        self.episode_buffer[11] += copy.deepcopy(current_index)
        self.episode_buffer[12] += copy.deepcopy(node_padding_mask)
        self.episode_buffer[13] += copy.deepcopy(edge_padding_mask)
        self.episode_buffer[14] += copy.deepcopy(edge_mask)

    def run_episode(self, curr_episode):
        done = False
        no_nbv = False
        # self.env.clear_trajectory()

        observations = self.get_observations()
        # obervations 在地图漂移的情况下会是None
        # 这个标志位的逻辑是正常情况的话会变为False
        is_observation_null = True
        # for i in range(128):
        # 30步一个episode，这个地方根据地图大小动态调整
        while is_observation_null is True:
            if done or no_nbv:
                break
            if observations is None:
                # 环境重启，重新这一轮
                self.env.reset()
            for i in range(60):
                # 经验池是一条一条的
                if observations is None:
                    observations = self.get_observations()
                # 获取完还是None，重置
                # None一个是真的None，一个是里程计漂移，索引超限
                if observations is None:
                    is_observation_null = True
                    print("因观测重置")
                    break
                next_position = None
                action_index = None
                is_node_select_null = False
                try:
                    next_position, action_index = self.select_node(observations)
                except AttributeError:
                    is_node_select_null = True
                # 这种方式会出现脏经验
                if is_node_select_null is True:
                    is_observation_null = True
                    print("因选点重置")
                    break
                self.save_observations(observations)
                policy_center_frontier = self.select_closest_frontier(next_position)
                # 经验池保存动作
                self.save_action(action_index)
                # print("nextposition:{} {}", next_position[0], next_position[1])
                reward, done, self.robot_position, self.travel_dist, plan_status, no_nbv, object_reward = self.env.step(
                    policy_center_frontier,
                    next_position,
                    self.travel_dist)
                # 这里直接跳出会导致 经验池的尺寸出问题
                # if reward < -10000:
                #     break
                # 在nbv异常终止的情况下，仅仅重启nbv不跳出
                if no_nbv:
                    reset_msg = Int8()
                    reset_msg.data = 1
                    self.reset_nbv_pub.publish(reset_msg)
                    time.sleep(2)
                    no_nbv = False
                start_time = timeit.default_timer()
                self.update_center_frontier = True
                # 簇中心点不再出现
                mutex = True
                # 重启后仍然获取不到边界中心，跳出，重置环境后再次重启nbv
                while self.update_center_frontier is True and mutex is True:
                    end_time = timeit.default_timer()
                    execution_time = end_time - start_time
                    if execution_time > 5:
                        # 重新唤醒nbv
                        # 跳出循环
                        print("超时，获取不到边界中心")
                        no_nbv = True
                        mutex = False
                if no_nbv is True and self.env.explored_area > 0.95 and done is False:
                    done = True
                    reward += 150
                # 连续规划失败2次，可以重启了，算是撞了
                if self.env.plan_filed_count >= 2:
                    reward -= 150
                    print("判断为撞了")
                self.save_reward_done(reward, done)
                print("reward:{}".format(reward))
                observations = self.get_observations()
                # 观测获取失败
                if observations is None:
                    break
                self.save_next_observations(observations)
                # plt.imshow(self.env.robot_belief, cmap='gray')
                # plt.show()
                # save a frame
                if self.save_image:
                    if not os.path.exists(gifs_path):
                        os.makedirs(gifs_path)
                    self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

                is_observation_null = False

                # 判断是否出现了规划失败，不再判断
                # 这里只负责跳出，修改值没有用，因为已经保存了
                if done or no_nbv or self.env.plan_filed_count >= 2:
                    break

        self.env.clear_trajectory()
        # save metrics
        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_area'] = self.env.explored_area
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['object_reward'] = object_reward

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)
        self.env.reset()
        # 重启nbv
        # if no_nbv is True:
        #     reset_msg = Int8()
        #     reset_msg.data = 1
        #     self.reset_nbv_pub.publish(reset_msg)
        #     time.sleep(2)
        # 先不按照条件了，可能更稳定
        reset_msg = Int8()
        reset_msg.data = 1
        self.env.reset_nbv_pub.publish(reset_msg)
        time.sleep(3)


    def work(self, currEpisode):
        self.run_episode(currEpisode)

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
