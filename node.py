import numpy as np


def one_hot_to_primes(one_hot_vector):
    primes = [3, 5, 7, 11, 13, 17]
    selected_primes = []
    product = 1

    for i in range(len(one_hot_vector)):
        if one_hot_vector[i] == 1:
            selected_primes.append(primes[i])
            product *= primes[i]
    if product == 1:
        product = 0
    return product


def number_to_one_hot(number):
    primes = [3, 5, 7, 11, 13, 17]
    one_hot_vector = [0] * len(primes)
    if number == 1:
        return one_hot_vector

    factors = []

    for prime in primes:
        while number % prime == 0:
            factors.append(prime)
            number //= prime

    for prime in factors:
        if prime in primes:
            index = primes.index(prime)
            one_hot_vector[index] = 1

    return one_hot_vector


class Node():
    def __init__(self, coords, frontiers, robot_belief):
        self.coords = coords
        # 节点级别的可观测边界
        self.observable_frontiers = []
        self.w = -5
        self.object_value = 0
        self.sensor_range = 80
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_w(self, value):
        self.w = value

    def set_object_value(self, one_hot):
        self.object_value = one_hot_to_primes(one_hot)

    def initialize_observable_frontiers(self, frontiers, robot_belief):
        # 这里有个重大问题就是我们拿到的都是map坐标系下的，但是算法中迭代的是像素坐标系下的
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.sensor_range - 10]
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_belief)
            if not collision:
                self.observable_frontiers.append(point)

    def get_node_utility(self):
        return len(self.observable_frontiers)

    def update_observable_frontiers(self, observed_frontiers, new_frontiers, robot_belief):
        # remove observed frontiers in the observable frontiers
        # if observed_frontiers != []:
        if observed_frontiers.size != 0:
            observed_index = []
            for i, point in enumerate(self.observable_frontiers):
                if point[0] + point[1] * 1j in observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j:
                    observed_index.append(i)
            for index in reversed(observed_index):
                self.observable_frontiers.pop(index)

        # add new frontiers in the observable frontiers
        # if new_frontiers != []:
        if new_frontiers.size != 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.sensor_range - 10]
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.append(point)

        self.utility = self.get_node_utility()
        if self.utility <= 2:
            self.utility = 0
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0
        self.zero_utility_node = True

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
