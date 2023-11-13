import copy

import imageio
import numpy as np
import subprocess
import os
import rospy
from nav_msgs.srv import GetMap
from rospy import Duration
from os import path
import time
import timeit
import math
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int8
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import create_cloud_xyz32, PointField
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalStatusArray, GoalID
from squaternion import Quaternion
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from jsk_recognition_msgs.msg import BoundingBoxArray

from graph_generator import *
from node import *
from skimage.measure import block_reduce

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


def convert_map_pixel(coord_value):
    return np.floor((coord_value - (-10) - 0.05 / 2) / 0.05)


def convert_pixel_map(coord_value):
    return (coord_value * 0.05) + (-10) + 0.05 / 2


# 检查以当前位置为中心的30x30范围内是否有连续的10个以上的值为0的点
def check_continuous_zeros(map, current_x, current_y):
    search_radius = 20  # 25是50x50范围的一半
    continuous_zeros_threshold = 10  # 连续0的阈值
    total_num_over_10 = 0
    visited = np.zeros([384, 384])

    left_range = current_x - search_radius
    right_range = current_x + search_radius
    up_range = current_y + search_radius
    bottom_range = current_y - search_radius
    for i in range(-search_radius, search_radius + 1):
        for j in range(-search_radius, search_radius + 1):
            x = int(current_x + i)
            y = int(current_y + j)

            if x < 0 or x >= map.shape[0] or y < 0 or y >= map.shape[1] or int(visited[x, y]) == 1:
                continue
            visited[x, y] = 1
            # 1为障碍占据
            if map[x, y] == 1:
                length = dfs_search(map, visited, x, y, left_range, right_range, up_range, bottom_range)
                if length >= 3:
                    total_num_over_10 += 1
    return True if total_num_over_10 >= 3 else False


def dfs_search(map, visited_table, current_x, current_y, left, right, up, bottom):
    # 34x34 dfs
    visited_table[current_x, current_y] = 1
    ret_nums = 0
    if left <= current_x <= right and bottom <= current_y <= up:
        # 经过实际检验，4x4太大了，只搜索一个单位的邻域即可
        for m in range(-2, 3):
            for n in range(-2, 3):
                if 0 <= current_x + m < map.shape[0] and 0 <= current_y + n < map.shape[1]:
                    if map[current_x + m, current_y + n] == 1 and visited_table[current_x + m, current_y + n] == 0:
                        ret_nums += dfs_search(map, visited_table, current_x + m,
                                               current_y + n, left, right, up, bottom) + 1
        return ret_nums
    else:
        return 0


# 统计目标点为中心的40x40区域的像素数量，并判断值为0的数量占比是否低于90%
def analyze_target_area(map, target_x, target_y):
    target_radius = 20  # 20是40x40范围的一半
    zero_threshold = 90

    zero_count = 0
    total_count = 0

    for i in range(-target_radius, target_radius + 1):
        for j in range(-target_radius, target_radius + 1):
            x = int(target_x + i)
            y = int(target_y + j)

            # 确保索引在合法范围内
            if 0 <= x < map.shape[0] and 0 <= y < map.shape[1]:
                total_count += 1
                if map[x, y] == 255:
                    zero_count += 1

    zero_percentage = zero_count / total_count * 100

    return zero_percentage < zero_threshold


class GazeboEnv:
    def __init__(self, map_index, k_size=20, plot=False, test=False):
        self.test = test

        # param
        self.explored_area = 0
        self.resolution = 4
        self.object_dist_matrix = None
        self.last_frontier_object = np.array([6, 1]) * 0
        self.sensor_range = 80
        self.plot = plot
        self.frame_files = []
        self.xPoints = None
        self.yPoints = None
        self.trajectory_points = []

        # initialize graph generator
        self.graph_generator = Graph_generator(map_size=np.array([384, 384]), sensor_range=self.sensor_range,
                                               k_size=k_size, plot=plot)
        # 放到begin中初始化
        # self.graph_generator.route_node.append(np.array([convert_map_pixel(-3.5), convert_map_pixel(-3.6)]))
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.object_value = None, None, None, None, None
        self.frontiers = None
        self.observed_frontiers = copy.deepcopy(self.frontiers)
        self.center_frontiers = None
        self.center_policy_frontier = None
        self.policy_object = np.zeros([6, 1]) * 0
        self.last_center_frontier = np.array([0.0, 0.0])
        self.last_object = copy.deepcopy(self.policy_object)
        self.bbox_detect_object = np.zeros([6, 1]) * 0
        self.bbox_detect_area = np.full((6, 2, 2), np.inf)
        self.last_detect_object = copy.deepcopy(self.bbox_detect_object)
        self.last_detect_area = copy.deepcopy(self.bbox_detect_area)
        self.robot_belief = np.ones((384, 384)) * 127

        # 连续规划失败基本是陷入了不可逆的困境
        self.plan_filed_count = 0
        self.move_plan_filed = False
        # flag均以锁的形式使用
        self.update_belief_flag = True
        self.update_position_flag = True
        # 获取导航标志需要与目标到达情况配合使用
        self.update_nav_status_flag = False
        self.update_bbox_flag = False
        # 是否有目标也要单独拉出来使用
        self.bbox_having_flag = False
        self.update_frontier_flag = True
        self.robot_position = None
        # 是否到达目标点也要单独拉出来使用
        self.goal_arrive_flag = False
        # old map
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.downsampled_belief = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "turtlebot3_waffle"

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)

        # Set up the ROS publishers and subscribers
        self.reset_nbv_pub = rospy.Publisher("/reset_nbv", Int8, queue_size=1)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.gmapping_reset_pub = rospy.Publisher("/reset_gmapping", Int8, queue_size=1)
        # 全局路径规划
        self.make_plan_service = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)
        # 画出轨迹的节点句柄
        self.marker_pub = rospy.Publisher("trajectory_marker", Marker, queue_size=1)
        self.goal_cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        self.odom = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.frontiers_sub = rospy.Subscriber(
            "/frontiers_pixel", Marker, self.get_frontier_callback, queue_size=1
        )
        self.goal_arrive_sub = rospy.Subscriber(
            "/move_base/status", GoalStatusArray, self.goal_arrive_callback
        )
        self.filter_box_sub = rospy.Subscriber(
            "/boxes_filtered", BoundingBoxArray, self.box_array_callback
        )
        self.map_sub = rospy.Subscriber(
            "/map", OccupancyGrid, self.map_callback
        )
        # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
        # range as state representation
        self.begin()
        # source code in here to controller plo2t

    def cal_path_plan_dist(self, start_point, goal_point):
        # 创建起始点和目标点
        start_pose = PoseStamped()
        start_pose.pose.position.x = convert_pixel_map(start_point[0])
        start_pose.pose.position.y = convert_pixel_map(start_point[1])
        start_pose.pose.position.z = 0.0
        start_pose.pose.orientation.w = 1.0
        start_pose.header.frame_id = "map"

        goal_pose = PoseStamped()
        goal_pose.pose.position.x = convert_pixel_map(goal_point[0])
        goal_pose.pose.position.y = convert_pixel_map(goal_point[1])
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        goal_pose.header.frame_id = "map"
        # 创建路径规划请求
        path_request = GetPlan()
        path_request.start = start_pose
        path_request.goal = goal_pose
        path_request.tolerance = 0.1  # 设置容忍度
        path_response = self.make_plan_service(path_request.start, path_request.goal, path_request.tolerance)
        path = path_response.plan
        path_length = 0.0
        for i in range(1, len(path.poses)):
            x1 = convert_map_pixel(path.poses[i - 1].pose.position.x)
            y1 = convert_map_pixel(path.poses[i - 1].pose.position.y)
            x2 = convert_map_pixel(path.poses[i].pose.position.x)
            y2 = convert_map_pixel(path.poses[i].pose.position.y)
            segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            path_length += segment_length
        return path_length, path.poses

    def backward_step(self):
        reverse_cmd = Twist()
        reverse_cmd.linear.x = -0.2  # 负线性速度表示后退
        self.vel_pub.publish(reverse_cmd)
        time.sleep(0.5)
        stop_cmd = Twist()
        self.vel_pub.publish(stop_cmd)

    def clear_cost_map(self):
        msg = Int8()
        msg.data = 2
        self.gmapping_reset_pub.publish(msg)

    def calculate_angle_to_target(self, current_pose, target_pose):
        # 计算当前位置到目标点的向量
        dx = target_pose[0] - convert_pixel_map(current_pose[0])
        dy = target_pose[1] - convert_pixel_map(current_pose[1])

        # 计算向量与地图X轴的夹角（假设地图X轴为正方向）
        yaw = math.atan2(dy, dx)

        # 将夹角转化为四元数
        quaternion = quaternion_from_euler(0, 0, yaw)

        return quaternion

    def send_goal(self, map_x, map_y):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = map_x
        goal.pose.position.y = map_y
        goal.pose.position.z = 0.0
        # 计算目标点与当前位置点之间的旋转四元数
        rotation_quaternion = self.calculate_angle_to_target(self.robot_position, np.array([map_x, map_y]))
        # 将计算的四元数赋值给目标点的位姿
        goal.pose.orientation.x = rotation_quaternion[0]
        goal.pose.orientation.y = rotation_quaternion[1]
        goal.pose.orientation.z = rotation_quaternion[2]
        goal.pose.orientation.w = rotation_quaternion[3]
        goal.header.stamp = rospy.Time.now()
        # print("发布目标点{}{}", map_x, map_y)
        self.goal_publisher.publish(goal)

    def goal_cancel(self):
        # print("取消目标点")
        goal_id = GoalID()
        self.goal_cancel_pub.publish(goal_id)

    def map_callback(self, msg):
        if self.update_belief_flag is True:
            # print("更新局部地图")
            belief = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            belief[belief == -1] = 127
            belief[(belief >= 0) & (belief <= 19.6)] = 255
            belief[(belief >= 65) & (belief <= 100)] = 1  # 一开始搞错了，搞成了0
            self.robot_belief = belief
            self.update_belief_flag = False

    # def box_array_callback(self, msg):
    #     if msg.boxes is not None and self.center_policy_frontier is not None:
    #         self.cpf_object = np.zeros((11, 1))
    #         if len(msg.boxes) != 0:
    #             for bbox in msg.boxes:
    #                 bbox_x = convert_map_pixel(bbox.pose.position.x)
    #                 bbox_y = convert_map_pixel(bbox.pose.position.y)
    #                 dist = np.linalg.norm(self.center_policy_frontier - np.array([bbox_x, bbox_y]))
    #                 # 修改参数
    #                 if dist < 0.5 / 0.05:
    #                     self.cpf_object[bbox.label] = 1
    #                     # print("边界点出现了语义目标")
    #                     # print(self.cpf_object)

    def box_array_callback(self, msg):
        # 目标不一定有，所以只检测一次，即使没有也关闭检测
        if self.update_bbox_flag is True:
            if msg.boxes is not None:
                self.last_detect_object = copy.deepcopy(self.bbox_detect_object)
                self.last_detect_area = copy.deepcopy(self.bbox_detect_area)
                # 每个step都有可能出现新目标
                self.bbox_detect_object = np.zeros([6, 1]) * 0
                self.bbox_detect_area = np.full((6, 2, 2), np.inf)
                if len(msg.boxes) != 0:
                    self.bbox_having_flag = True
                    for bbox in msg.boxes:
                        print("发现目标")
                        bbox_x_center = convert_map_pixel(bbox.pose.position.x)
                        bbox_y_center = convert_map_pixel(bbox.pose.position.y)
                        bbox_x_length = bbox.dimensions.x / 0.05
                        bbox_y_length = bbox.dimensions.y / 0.05
                        self.bbox_detect_object[bbox.label] = 1
                        self.bbox_detect_area[bbox.label] = np.array([(bbox_x_center, bbox_y_center),
                                                                      (bbox_x_length, bbox_y_length)])
            self.update_bbox_flag = False

    def goal_arrive_callback(self, goal_status):
        # self.goal_arrive_flag = False
        if self.update_nav_status_flag is True:
            if len(goal_status.status_list) > 0:
                # 获取最新的导航状态
                # print("获取导航状态")
                status = goal_status.status_list[-1]
                if status.status == 3:  # status 3 表示导航成功完成
                    # print("导航成功")
                    self.goal_arrive_flag = True
                elif status.status == 4: # status 4 表示规划失败
                    self.move_plan_filed = True
        # else:
        #     print("获取导航状态禁用中")

    def get_frontier_callback(self, frontier_marker):
        if self.update_frontier_flag is True:
            if len(frontier_marker.points) > 0:
                x_coordinates = [point.x for point in frontier_marker.points]
                y_coordinates = [point.y for point in frontier_marker.points]
                # Create a 2D NumPy array from the x and y coordinates
                self.frontiers = np.array([x_coordinates, y_coordinates]).T
                self.update_frontier_flag = False

    def update_robot_belief(self, robot_belief):
        self.old_robot_belief = robot_belief
        # 更新完告知地图回调，更新局部地图
        time.sleep(2)
        self.update_belief_flag = True

    def begin(self):
        # 检测一下目标
        self.update_bbox_flag = True
        while self.update_bbox_flag:
            pass
        # 等待更新机器人的状态
        start_time_1 = timeit.default_timer()
        while self.robot_position is None or self.robot_belief is None or self.frontiers is None:
            end_time = timeit.default_timer()
            if end_time - start_time_1 > 5:
                # 先不按照条件了，可能更稳定
                reset_msg = Int8()
                reset_msg.data = 1
                self.reset_nbv_pub.publish(reset_msg)
                time.sleep(3)
        if self.plot:
            # initialize the route
            self.xPoints = [self.robot_position[0]]
            self.yPoints = [self.robot_position[1]]
        self.observed_frontiers = copy.deepcopy(self.frontiers)
        self.update_robot_belief(self.robot_belief)
        # downsampled belief has lower resolution than robot belief
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        # self.frontiers = self.find_frontier()
        # frontier is automatically updated by callback
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.graph_generator.route_node.append(
            self.robot_position
        )
        # self.node_coords, self.graph, self.node_utility, self.guidepost = self.graph_generator.generate_graph(
        #     self.robot_position, self.robot_belief, self.frontiers)
        # 生成图添加目标的观测
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.object_value = self.graph_generator.generate_graph(
            self.robot_position, self.robot_belief, self.frontiers, self.bbox_detect_object, self.bbox_detect_area,
            self.bbox_having_flag)
        # 用完记得重置
        self.bbox_having_flag = False
        self.gen_relation_matrix()

    def odom_callback(self, od_data):
        # 画轨迹
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.LINE_STRIP
        marker.ns = "rviz_traj"
        marker.id = 0
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        # 消除警告
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(1.0)
        position = od_data.pose.pose.position
        self.trajectory_points.append(position)
        marker.points = self.trajectory_points
        marker.header.stamp = rospy.Time.now()
        self.marker_pub.publish(marker)
        # 更新定位信息
        if self.update_position_flag is True:
            # print("更新定位")
            # Extract the map coordinates
            x_map = od_data.pose.pose.position.x
            y_map = od_data.pose.pose.position.y
            # Inverse transformation to pixel coordinates
            tx = convert_map_pixel(x_map)
            ty = convert_map_pixel(y_map)
            self.robot_position = np.array([tx, ty])
            self.update_position_flag = False

    # 清除轨迹
    def clear_trajectory(self):
        # self.trajectory_points = []
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.DELETE
        marker.ns = "rviz_traj"
        marker.id = 0
        # marker.scale.x = 0.1
        # marker.color.r = 0.0
        # marker.color.g = 0.0
        # marker.color.b = 0.0
        # marker.color.a = 0.0
        # marker.points = self.trajectory_points
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = Duration(1.0)
        self.marker_pub.publish(marker)
        self.trajectory_points = []

    # 策略已经生成所选择的边界点，边界点的集合应该在状态空间中
    def step(self, policy_center_frontiers, next_position, travel_dist):
        self.center_policy_frontier = policy_center_frontiers
        # 此时的机器人位置是未导航之前的
        # 修改dist计算方式
        # dist = np.linalg.norm(next_position - self.robot_position)
        dist, path_points = self.cal_path_plan_dist(self.robot_position, next_position)
        travel_dist += dist
        goal_x = convert_pixel_map(next_position[0])
        goal_y = convert_pixel_map(next_position[1])
        # 用于计算奖励用，计算奖励时position已经是导航完成的
        start_position = copy.deepcopy(self.robot_position)
        self.send_goal(goal_x, goal_y)
        # 开启导航状态检测
        self.update_nav_status_flag = True
        # 选择边界点的拷贝当前边界点(self.robot_position)，这个应该等到导航结束再行拷贝，否则会将中间点覆盖
        start_time = time.time()
        plan_failed = False
        # while self.goal_arrive_flag is False:
        #     # 加速之后规划路径变快
        #     # if time.time() - start_time > 20:
        #     if time.time() - start_time > 8:
        #         # 规划失败，关闭导航状态检测
        #         self.goal_cancel()
        #         # print("规划失败")
        #         self.update_nav_status_flag = False
        #         # 规划失败后，是否重新获取定位呢，不用在这里，在reset里重置Gazebo后再获取
        #         # travel_dist -= dist  # 不用减去
        #         # return -100, False, self.robot_position, travel_dist
        #         plan_failed = True
        #         # 后退一步
        #         self.backward_step()
        #         # 更改实验后好像不需要清除代价地图了，看看效果
        #         # 清除代价地图
        #         # self.clear_cost_map()
        #         self.plan_filed_count += 1
        #         break

        # 新方法，直接获取状态码
        # 仍然需要超时保障
        while self.goal_arrive_flag is False and self.move_plan_filed is False:
            if time.time() - start_time > 8:
                    # 规划失败，关闭导航状态检测
                    self.goal_cancel()
                    self.update_nav_status_flag = False
                    plan_failed = True
                    # 后退一步
                    self.backward_step()
                    self.plan_filed_count += 1
                    break

        if self.move_plan_filed:
            self.goal_cancel()
            self.update_nav_status_flag = False
            plan_failed = True
            self.backward_step()
            self.plan_filed_count += 1

        if plan_failed is False:
            # 规划成功，清空
            self.plan_filed_count = 0

        self.goal_arrive_flag = False
        self.move_plan_filed = False
        # 关闭导航状态检测
        self.update_nav_status_flag = False
        # robot_position = next_position 源码中是直接改变状态空间，因为不需要导航，所以此处应该是导航，并非直接进行赋值替换
        # 导航结束后重新获取定位，开放flag，也不是直接赋值
        self.update_position_flag = True  # 替代robot_position = next_position
        # time_cost = 0
        # while self.goal_arrive_flag is False:
        #     if time_cost > 5:
        #         # 规划失败，奖励需要惩罚
        #         pass
        # 下面的程序应该保证定位和belief是准确的，刚打开标志位立马使用是不准的
        self.update_robot_belief(self.robot_belief)  # 导航完了，更新区域
        start_time_1 = timeit.default_timer()
        while self.update_position_flag is True:
            pass
        end_time = timeit.default_timer()
        execution_time = end_time - start_time_1
        # print("更新位置信息成功，用时{}s,当前位置({},{})".format(execution_time, self.robot_position[0], self.robot_position[1]))
        # self.update_belief_flag = True # update_robot_belief方法已经作了这个工作
        # 规划失败，但是没有进入条件
        # if np.linalg.norm(self.robot_position - next_position) > 6:
        #     self.goal_cancel()
        #     plan_failed = True

        start_time_1 = timeit.default_timer()
        while self.update_belief_flag is True:
            pass
        end_time = timeit.default_timer()
        execution_time = end_time - start_time_1
        # print("更新belief信息成功，用时{}s".format(execution_time))
        # 导航完了，更新边界
        self.observed_frontiers = copy.deepcopy(self.frontiers)
        self.update_frontier_flag = True
        start_time_1 = timeit.default_timer()
        no_nbv = False
        while self.update_frontier_flag is True:
            # 程序卡在了这里，设置个超时，但是超时返回什么，最好的方式就是不异常进入
            run_time = timeit.default_timer()
            if run_time - start_time_1 > 5:
                # no_nbv =
                print("step获取不到边界")
                # 重启nbv，不应该直接返回，因为有的时候地图出问题，会导致nbv出问题。而不是探索完了，先直接返回吧，但是还是要区分nbv和done
                return 0, False, self.robot_position, travel_dist, False, True, 0
        end_time = timeit.default_timer()
        execution_time = end_time - start_time_1
        # print("更新边界信息成功，用时{}s".format(execution_time))
        # 检测目标的存在性
        self.update_bbox_flag = True
        start_time_1 = timeit.default_timer()
        while self.update_bbox_flag is True:
            pass
        end_time = timeit.default_timer()
        execution_time = end_time - start_time_1
        # print("检测目标信息成功，用时{}s".format(execution_time))
        # route添加方式变更
        # self.graph_generator.route_node.append(self.robot_position)
        # next_node_index = self.find_index_from_coords(self.robot_position)
        # self.graph_generator.nodes_list[next_node_index].set_visited()
        for i in range(len(path_points)):
            current_point = np.array([path_points[i].pose.position.x, path_points[i].pose.position.y])
            self.graph_generator.route_node.append(current_point)
            next_node_index = self.find_index_from_coords(current_point)
            self.graph_generator.nodes_list[next_node_index].set_visited()
        # 最后再加上
        self.graph_generator.route_node.append(self.robot_position)
        next_node_index = self.find_index_from_coords(self.robot_position)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        # 探索率如何计算是个有待商榷的问题
        self.explored_area = self.evaluate_exploration_area()
        # 将下一个机器人的位置点与当前位置的距离计算出，并将决策选择的下一点加入图的route node
        # 计算奖励，考虑语义目标相关性的分数，鼓励靠近选择的边界点
        # update the graph
        # 注意边界的用处，因为我们的边界是确定的
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.object_value = self.graph_generator.update_graph(
            self.robot_position, self.robot_belief, self.old_robot_belief, self.frontiers, self.observed_frontiers,
            self.bbox_detect_object, self.bbox_detect_area, self.bbox_having_flag)
        # 重置标志位
        self.bbox_having_flag = False
        # 在begin中初始化
        # self.observed_frontiers = copy.deepcopy(self.frontiers)
        # 边界不用传了，上一次观测的边界已经记录在了self.observed中，self.froniters也已经更新
        reward, object_reward = self.calculate_reward(dist, self.policy_object, policy_center_frontiers, next_position,
                                                      start_position,
                                                      travel_dist)
        done = self.check_done()
        if self.plot:
            self.xPoints.append(self.robot_position[0])
            self.yPoints.append(self.robot_position[1])
        # # 边界中心未发生改变，证明没有新边界可以发现了，完成, 不好考虑，因为这样动作幅度小的话也会导致边界中心不发生变化
        # if np.array_equal(self.center_policy_frontier, self.last_center_frontier):
        #     done = True
        # self.last_center_frontier = copy.deepcopy(self.center_policy_frontier)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        if done:
            reward += 150  # a finishing reward
        #
        if plan_failed:
            reward -= 30  # plan failed reward
        return reward, done, self.robot_position, travel_dist, plan_failed, no_nbv, object_reward

    def evaluate_exploration_area(self):
        # 由于我们不存在groud_truth，所以没有真正的探索率，用探索面积代替
        area = np.sum(self.robot_belief == 255)
        # groud_truth 既然是特定场景，那么可以获取
        area = np.sum(self.robot_belief == 255) / 33376
        return area

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def calculate_reward(self, dist, object_vector, policy_center_frontiers, next_position, start_position,
                         travel_dist):
        reward = 0
        # reward -= dist / 64  # dist为当前点和下一视点的距离
        reward -= travel_dist / 500
        # check the num of observed frontiers
        frontiers_to_check = self.frontiers[:, 0] + self.frontiers[:, 1] * 1j
        pre_frontiers_to_check = self.observed_frontiers[:, 0] + self.observed_frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        # 这里不应该是旧边界减去共有边界，应该是新边界减去共有边界才是新发现的边界
        # pre_frontiers_num = pre_frontiers_to_check.shape[0]
        # 仿真环境中的边界噪声严重，感觉奖励可以用新自由区域等价替换
        pre_frontiers_num = frontiers_to_check.shape[0]
        # delta_num = pre_frontiers_num - frontiers_num
        delta_num = np.sum(self.robot_belief == 255) - np.sum(self.old_robot_belief == 255)
        if delta_num < 300:
            delta_num = 0

        # 计算语义目标关系得分
        score = self.match_object_score() * 20
        # 计算与选择的边界簇中心距离，引导机器人靠近选择的边界中心
        # 簇中心选择的好坏需要任务的完成情况得分来评判，毕竟簇中心的选择是策略的输出
        # self.robot_postion 是已经导航完毕的机器人所在点
        policy_dist = np.linalg.norm(policy_center_frontiers - self.robot_position)  # 当前位置与所选择边界点之间的距离
        # score_policy_dist = - policy_dist / 64 # 太粗暴，不能只是惩罚
        is_current_cluster = check_continuous_zeros(self.robot_belief, self.robot_position[0], self.robot_position[1])
        is_target_free = analyze_target_area(self.robot_belief, next_position[0], next_position[1])
        start2cfp = np.linalg.norm(start_position - policy_center_frontiers)
        # score_policy_dist = (start2cfp - policy_dist) / policy_dist * 0.7
        # 换一种思路，不用绝对欧式距离占比，而用有效靠近比例
        # score_policy_dist = (start2cfp - policy_dist) / np.linalg.norm(start_position - next_position)
        # 以上方式也是有问题的表达方式

        path_length_start2cpf, _ = self.cal_path_plan_dist(start_position, policy_center_frontiers)
        path_length_position2cpf, _ = self.cal_path_plan_dist(self.robot_position, policy_center_frontiers)
        path_length_start2next, _ = self.cal_path_plan_dist(start_position, next_position)
        if path_length_start2next == 0:
            score_policy_dist = 0
        else:
            score_policy_dist = (path_length_start2cpf - path_length_position2cpf) / path_length_start2next
        if score_policy_dist <= 0.15:
            score_policy_dist = (score_policy_dist - 1) * 5
        alpha = 0.0
        if is_current_cluster is True and is_target_free is True:
            # 惩罚，容易撞
            alpha = -1.0
            pass
        elif is_current_cluster is True and is_target_free is False:
            # 严重惩罚，非常恶劣
            alpha = -1.2
            pass
        elif is_current_cluster is False and is_target_free is True:
            # 奖励，鼓励
            alpha = 0.8 if score_policy_dist > 0 else -1.0
            pass
        elif is_current_cluster is False and is_target_free is False:
            # 奖励，鼓励，更大一点，趋近于发现目标
            alpha = 1.0 if score_policy_dist > 0 else -1.5
            pass
        # action_dist_score = alpha * dist / 38
        # score_policy_dist = - policy_dist / 64  # 这样会一直得到负奖励，不能起到引导作用
        # 采用当前点到下一视点到所选边界的距离差对于距离的占比来计算奖励
        # reward += delta_num / 60
        # 即使不需要边界发现值了，也需要设置
        area_find_score = delta_num / 100
        # 还有一点就是两个选择的边界中心是不是会距离很远，根本难以匹配，所以最好还是障碍物语义应该在节点上，用节点去判断
        reward += score  # 观察这里是不是也是太粗暴了，有点难以观察
        # reward += action_dist_score
        reward += area_find_score
        reward += score_policy_dist
        # reward += area_find_score
        # reward += - dist / 38
        return reward, score

    def match_object_score(self):
        # 计算当前策略选择簇中心最近的上一次的簇中心语义向量，如果两个簇中心距离足够近则匹配加分
        # 计算匹配的关系数量
        matching_count = np.sum(
            np.logical_and(self.last_detect_object, self.bbox_detect_object) * self.object_dist_matrix)
        if matching_count != 0:
            print("有语义得分")
            print(self.last_detect_object)
            print(self.bbox_detect_object)
        return matching_count

    def reset(self):
        self.clear_trajectory()
        self.bbox_detect_object = np.zeros([6, 1]) * 0
        self.bbox_detect_area = np.full((6, 2), np.inf)
        object_state = self.set_self_state
        object_state.pose.position.x = -3.5
        object_state.pose.position.y = -3.6
        object_state.pose.position.z = 0.0
        quaternion = Quaternion.from_euler(1.0, 0.0, 0.0)
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)
        self.update_position_flag = True
        while self.update_position_flag is True:
            pass
        self.update_belief_flag = True
        while self.update_belief_flag is True:
            pass
        msg = Int8()
        msg.data = 1
        rospy.sleep(3.0)
        self.gmapping_reset_pub.publish(msg)
        rospy.sleep(2.0)

    def gen_relation_matrix(self):
        # 定义目标类别和关系
        # target_classes = ["table_breakfast", "sofa", "desk", "chair_office", "table", "table_stable",
        #                   "file_cabinet", "chair_breakfast", "table_large", "trash_bin", "book_shelf"]
        target_classes = ["chair_breakfast", "table_breakfast", "chair_office", "desk", "table", "sofa"]

        # 初始化关系矩阵
        num_classes = len(target_classes)
        relation_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # 定义目标之间的关系
        # relations = {
        #     ("table_breakfast", "chair_breakfast"): 1,
        #     ("sofa", "table"): 1,
        #     ("sofa", "table_stable"): 1,
        #     ("desk", "chair_office"): 1,
        #     ("file_cabinet", "desk"): 1,
        #     ("file_cabinet", "chair_office"): 1,
        #     ("desk", "desk"): 1,
        #     ("sofa", "sofa"): 1
        # }
        relations = {
            ("desk", "sofa"): 1,
            ("chair_breakfast", "table_breakfast"): 1,
            ("desk", "chair_office"): 1,
            ("desk", "desk"): 1,
            ("sofa", "sofa"): 1
        }

        # 填充关系矩阵
        for i in range(num_classes):
            for j in range(num_classes):
                if (target_classes[i], target_classes[j]) in relations:
                    relation_matrix[i, j] = relation_matrix[j, i] = relations[(target_classes[i], target_classes[j])]
        self.object_dist_matrix = relation_matrix

    def check_done(self):
        done = False
        # 手动建图获得的自由区域的像素点数量为33376
        # 在原本的设计中，训练阶段看节点效用，验证阶段看节点效用以及探索率
        # if self.test and np.sum(self.robot_belief == 255) >= int(33376 * 0.99):
        if np.sum(self.robot_belief == 255) >= int(33376 * 0.99):
            done = True
        elif np.sum(self.node_utility) == 0:
            done = True
        return done

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        # plt.ion()
        plt.cla()
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis((0, 384, 384, 0))
        for i in range(len(self.graph_generator.x)):
            plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan',
                     zorder=1)  # plot edges will take long time
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.node_utility, zorder=5)
        object_coords = [self.node_coords[i] for i in np.where(self.object_value != 0)[0]]
        object_coords = np.array(object_coords)
        if object_coords.__len__() != 0:
            # plt.scatter(object_coords[:, 0], object_coords[:, 1], c=self.object_value[self.object_value != 0].astype(int), zorder=6)
            # plt.plot(object_coords[:, 0], object_coords[:, 1], c='r', linewidth=1, zorder=1)
            values = self.object_value[self.object_value != 0].astype(int)
            colors = plt.cm.viridis(values / values.max())  # 根据值生成颜色，这里使用了viridis colormap
            plt.scatter(object_coords[:, 0], object_coords[:, 1], c=colors, marker='s', s=80, zorder=6)
            print("绘制目标点")
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=8)
        plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)
        # plt.pause(0.1)
        plt.suptitle('Explored area: {}  Travel distance: {:.4g}'.format(self.explored_area, travel_dist))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)

# if __name__ == '__main__':
#     env = GazeboEnv(1)
#     env.update_frontier_flag = True
#     while env.update_frontier_flag is True:
#         pass
#     env.observed_frontiers = copy.deepcopy(env.frontiers)
#     env.update_frontier_flag = True
#
#     while True:
#         pass
#     x = 1
#     y = 1
#     while True:
#         env.send_goal(x, y)
