import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

from node import Node
from graph import Graph, a_star


class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        # 添加节点新发现边界点
        self.new_node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        # 将位姿预判尝试加入
        self.w = []
        # 384x384
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.sensor_range = sensor_range
        self.uniform_points = self.generate_uniform_points()
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.guidepost = None
        # 检测到的目标
        self.object_value = None

    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []
        self.w = []

    def edge_clear(self, coords):
        node_index = str(self.find_index_from_coords(self.node_coords, coords))
        self.graph.clear_edge(node_index)

    def check_object_range(self, coords, object_vector, object_area):
        # 拆解目标
        current_node_one_hot = np.zeros([6, 1])
        # 是否需要更新
        is_update_object = False
        # 检查当前坐标是否在area内
        for index, item in enumerate(object_vector):
            # 检测到目标
            if item == 1:
                area = object_area[index]
                max_x = area[0][0] + area[1][0] / 2
                max_y = area[0][1] + area[1][1] / 2
                min_x = area[0][0] - area[1][0] / 2
                min_y = area[0][1] - area[1][1] / 2
                if min_x <= coords[0] <= max_x and min_y <= coords[1] <= max_y:
                    is_update_object = True
                    current_node_one_hot[index] = 1
        return is_update_object, current_node_one_hot

    # object_vector是当前检测到的所有目标组成的one-hot向量，object_area为当前检测到的所有目标对应的object_nums x 2的向量
    def generate_graph(self, robot_location, robot_belief, frontiers, object_vector, object_area, bbox_having_flag):
        # get node_coords by finding the uniform points in free area
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]

        # add robot location as one node coords
        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)

        # generate the collision free graph
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        # calculate the utility as the number of observable frontiers of each node
        # save the observable frontiers to be reused
        self.node_utility = []
        self.object_value = []
        self.w = []
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_belief)
            if bbox_having_flag is True:
                is_update_object, current_node_one_hot = self.check_object_range(coords, object_vector, object_area)
                # 更新检测到的目标向量
                if is_update_object is True:
                    node.set_object_value(current_node_one_hot)
            self.nodes_list.append(node)
            utility = node.utility
            self.node_utility.append(utility)
            # 添加检测目标
            self.object_value.append([node.object_value])
        print(self.object_value)
        self.node_utility = np.array(self.node_utility)
        # 转换为数组
        self.object_value = np.array(self.object_value)

        # guidepost is a binary sign to indicate weather one node has been visited
        # 通过维护的route判断的
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        self.object_value = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j
        for node in self.route_node:
            # 将起始点加入
            index = np.argwhere(x.reshape(-1) == node[0] + node[1] * 1j)[0]
            self.guidepost[index] = 1

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost, self.object_value

    def update_graph(self, robot_position, robot_belief, old_robot_belief, frontiers, old_frontiers, object_vector,
                     object_area, bbox_having_flag):
        # add uniform points in the new free area to the node coords
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        self.new_node_coords = copy.deepcopy(new_node_coords)
        old_node_coords = copy.deepcopy(self.node_coords)
        self.node_coords = np.concatenate((self.node_coords, new_node_coords))
        # 防止重复
        self.node_coords = np.unique(self.node_coords, axis=0)

        # update the collision free graph
        # for coords in new_node_coords:
        #     self.find_k_neighbor(coords, self.node_coords, robot_belief)
        # dist_to_robot = np.linalg.norm(robot_position - old_node_coords, axis=1)
        # nearby_node_indices = np.argwhere(dist_to_robot <= 160)[:, 0].tolist()
        # for index in nearby_node_indices:
        #     coords = old_node_coords[index]
        #     self.edge_clear(coords)
        #     self.find_k_neighbor(coords, self.node_coords, robot_belief)

        self.edge_clear_all_nodes()
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

        # update the observable frontiers through the change of frontiers
        old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        observed_frontiers_index = np.where(
            np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
        new_frontiers_index = np.where(
            np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
        observed_frontiers = old_frontiers[observed_frontiers_index]
        new_frontiers = frontiers[new_frontiers_index]
        for node in self.nodes_list:
            if np.linalg.norm(node.coords - robot_position) > 2 * self.sensor_range:
                pass
            elif node.zero_utility_node is True:
                pass
            else:
                node.update_observable_frontiers(observed_frontiers, new_frontiers, robot_belief)

        for new_coords in new_node_coords:
            node = Node(new_coords, frontiers, robot_belief)
            # 检测到的目标不一定在新增区域
            # if bbox_having_flag is True: is_update_object, current_node_one_hot = self.check_object_range(
            # new_coords, object_vector, object_area) # 更新检测到的目标向量 if is_update_object is True:
            # node.set_object_value(current_node_one_hot) # 不清空，因为它是增量更新，不像效用是全局更新 # self.object_value.append(
            # node.object_value) self.object_value = np.concatenate([self.object_value, np.array([[
            # node.object_value]])])
            self.nodes_list.append(node)

        self.node_utility = []
        self.object_value = []
        for i, coords in enumerate(self.node_coords):
            utility = self.nodes_list[i].utility
            self.node_utility.append(utility)
            # if bbox_having_flag is True:
            #     is_update_object, current_node_one_hot = self.check_object_range(coords, object_vector, object_area)
            #     # 更新检测到的目标向量
            #     if is_update_object is True:
            #         self.nodes_list[i].set_object_value(current_node_one_hot)
            # 采用数组遍历的方式
            for object_index in range(len(object_vector)):
                if bbox_having_flag is True:
                    is_update_object, current_node_one_hot = self.check_object_range(coords, object_vector[object_index], object_area[object_index])
                    # 更新检测到的目标向量
                    if is_update_object is True:
                        self.nodes_list[i].set_object_value(current_node_one_hot)
            self.object_value.append([self.nodes_list[i].object_value])
            # self.object_value = np.concatenate([self.object_value, np.array([[node.object_value]])])
        self.node_utility = np.array(self.node_utility)
        # 添加，全局更新后，处理逻辑也相应变化
        self.object_value = np.array(self.object_value)

        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j
        for node in self.route_node:
            index = np.argwhere(x.reshape(-1) == node[0] + node[1] * 1j)
            self.guidepost[index] = 1

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost, self.object_value

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, 30).round().astype(int)
        y = np.linspace(0, self.map_y - 1, 30).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords - node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k < node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief):
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                dist = np.linalg.norm(start - end)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, dist)

            k += 1
        return neighbor_index_list

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        # 理解节点坐标系的构建的重要点
        # 制作一个连通图，去掉有碰撞的
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        # 膨胀障碍物
        # inf_robot_belief = self.inflation_robot_belief(robot_belief, radius=8)

        # 对于i节点丢失的bug进行重建模
        while True:
            for i, p in enumerate(X):
                for j, neighbour in enumerate(X[indices[i][:]]):
                    start = p
                    end = neighbour
                    # if not self.check_collision(start, end, robot_belief):
                    if not self.check_collision(start, end, robot_belief):
                        a = str(self.find_index_from_coords(node_coords, p))
                        b = str(self.find_index_from_coords(node_coords, neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j])

                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])
                    # else:
                    #     print(str(self.find_index_from_coords(node_coords, p)))
            if len(self.graph.edges) == len(self.graph.nodes):
                break
            print("节点丢失，重新更新")

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    # 定义一个函数来更新robot_belief数组
    def inflation_robot_belief(self, robot_belief, radius=4):
        # 复制原始数组以保持原始数据不变
        updated_belief = np.copy(robot_belief)

        # 获取数组的高度和宽度
        height, width = robot_belief.shape

        # 创建一个掩码，用于表示半径内的像素
        mask = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)

        # 填充掩码，将半径范围内的像素置为True
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if x ** 2 + y ** 2 <= radius ** 2:
                    mask[y + radius, x + radius] = True

        # 遍历整个数组
        for y in range(height):
            for x in range(width):
                # 如果当前像素点的值为1
                if robot_belief[y, x] == 1:
                    # 遍历半径范围内的像素
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            # 计算邻域像素的坐标
                            new_x, new_y = x + dx, y + dy
                            # 检查新坐标是否在数组范围内，并且在半径范围内
                            if (0 <= new_x < width and 0 <= new_y < height and
                                    mask[dy + radius, dx + radius]):
                                # 将邻域像素点置为1
                                updated_belief[new_y, new_x] = 1

        return updated_belief

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision

    def find_shortest_path(self, current, destination, node_coords):
        start_node = str(self.find_index_from_coords(node_coords, current))
        end_node = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(start_node), int(end_node), self.node_coords, self.graph)
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route
