import numpy as np
import heapq
from enum import IntEnum
from collections import deque
from itertools import chain
# import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import cv2  # pip install opencv-python
import imutils  # pip install imutils
# from fibonacci_heap_mod import Fibonacci_heap  # pip install fibonacci-heap-mod


class DistanceMode(IntEnum):
    lecture = 0
    paper = 1


class PixelNode(object):
    Directions = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)])

    def __init__(self, cost, pos, prev=None):
        self.state = 0  # 0 = initial, 1 = active, 2 = expanded
        self.cost = cost
        self.level = None  # for use in path-tree
        self.pos = pos
        self.prev = prev

    def __lt__(self, other):
        return self.cost - other.cost < 0

    def __hash__(self):
        return hash(self.pos)

    def __str__(self):
        return "(cost={}, state={}, cur={}, prev={})".format(self.cost, self.state, self.pos, self.prev)


class IScissor(object):

    def __init__(self, img=None):
        self.img = None
        self.orig_img = None
        self.dist_mode = DistanceMode.lecture
        self.w = None
        self.h = None
        self.cost_matrix = None
        self.use_snapping = False
        self.edges = None
        self.contours = []
        self.paths = []
        self.count = 0  # number of closed nodes
        self.max_level = 0  # maximum level of nodes, used in displaying path tree
        self.seed = None
        self.pq = None
        self.pixel_map = None
        # self.initialize_pixel_map()
        if img is not None:
            self.set_image(img)

    def _reset(self):
        self.cost_matrix = None
        self.contours = []
        self.paths = []
        self.count = 0  # number of closed nodes
        self.max_level = 0  # maximum level of nodes, used in displaying path tree
        self.seed = None
        self.pq = None
        self.pixel_map = None
        self.edges = imutils.auto_canny(self.img)

    def set_image(self, img):
        if img.dtype == 'float32':
            self.img = (img * 255).astype('uint8')
        else:
            self.img = np.copy(img)
        if self.img.shape[-1] > 3:
            # discard the alpha from RGBA
            self.img = self.img[:, :, :-1]
        self.h, self.w = self.img.shape[:2]
        self._reset()

    def set_dist_mode(self, mode):
        self.dist_mode = mode
        self._reset()

    def initialize_pixel_map(self):
        self.pixel_map = np.array([[PixelNode(np.inf, (j, i), None) for i in range(self.w)] for j in range(self.h)])
        self.pixel_map[self.seed].cost = 0
        self.pixel_map[self.seed].state = 1
        self.pixel_map[self.seed].level = 0
        self.pq = []
        self.count = 0

    def compute_cost(self):
        """
        Compute the cost of a given img, according to the distance mode.
        """
        if self.dist_mode == DistanceMode.lecture:
            return self._compute_cost_lecture_mode()
        elif self.dist_mode == DistanceMode.paper:
            return self._compute_cost_paper_mode()
        else:
            raise ValueError("Distance mode invalid. Please choose from lecture mode or paper mode.")

    def _compute_cost_lecture_mode(self):
        """
        Compute the cost of a given img, as in the project description.
        """
        D_matrix = np.zeros((*self.img.shape[:2], 8))

        # Note: To subtract from the pixel above the center is the same as translating the image *DOWN* one step and
        # subtract from the center. Same for other directions.
        D_matrix[:, :, 1] = np.sqrt(np.sum(np.square(
            np.abs(imutils.translate(self.img, 1, 0) - imutils.translate(self.img, 0, -1))) / 2,
                                           axis=2) / 3)
        D_matrix[:, :, 3] = np.sqrt(np.sum(np.square(
            np.abs(imutils.translate(self.img, 1, 0) - imutils.translate(self.img, 0, 1))) / 2,
                                           axis=2) / 3)
        D_matrix[:, :, 5] = np.sqrt(np.sum(np.square(
            np.abs(imutils.translate(self.img, -1, 0) - imutils.translate(self.img, 0, 1))) / 2,
                                           axis=2) / 3)
        D_matrix[:, :, 7] = np.sqrt(np.sum(np.square(
            np.abs(imutils.translate(self.img, -1, 0) - imutils.translate(self.img, 0, -1))) / 2,
                                           axis=2) / 3)

        D_matrix[:, :, 0] = np.sqrt(np.sum(np.square(
            np.abs((imutils.translate(self.img, 1, 0) + imutils.translate(self.img, 1, -1)) / 2
                   - (imutils.translate(self.img, -1, 0) + imutils.translate(self.img, -1, -1)) / 2)), axis=2) / 3)
        D_matrix[:, :, 2] = np.sqrt(np.sum(np.square(
            np.abs((imutils.translate(self.img, 0, 1) + imutils.translate(self.img, 1, 1)) / 2
                   - (imutils.translate(self.img, 0, -1) + imutils.translate(self.img, 1, -1)) / 2)), axis=2) / 3)
        D_matrix[:, :, 4] = np.sqrt(np.sum(np.square(
            np.abs((imutils.translate(self.img, -1, 0) + imutils.translate(self.img, -1, 1)) / 2
                   - (imutils.translate(self.img, 1, 0) + imutils.translate(self.img, 1, 1)) / 2)), axis=2) / 3)
        D_matrix[:, :, 6] = np.sqrt(np.sum(np.square(
            np.abs((imutils.translate(self.img, 0, 1) + imutils.translate(self.img, -1, 1)) / 2
                   - (imutils.translate(self.img, 0, -1) + imutils.translate(self.img, -1, -1)) / 2)), axis=2) / 3)

        maxD = np.max(D_matrix)
        D_matrix = maxD - D_matrix
        D_matrix[:, :, 1::2] = D_matrix[:, :, 1::2] * np.sqrt(2)

        return D_matrix

    def _compute_cost_paper_mode(self):
        """
        Compute the cost of a given img, as in the paper, except we used auto-canny for the binary edge detection part.
        """
        wz, wd, wg = 0.43, 0.43, 0.14
        fz = np.zeros((*self.img.shape[:2], 8))
        fd = np.zeros((*self.img.shape[:2], 8))
        fg = np.zeros((*self.img.shape[:2], 8))

        translator = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        ez = (255 - self.edges) / 255
        for i, t in enumerate(translator):
            fz[:, :, i] = imutils.translate(ez, *t)

        grayscale_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        Ix = (imutils.translate(grayscale_img, -1, 0) - grayscale_img).astype('float64')
        Iy = (imutils.translate(grayscale_img, 0, -1) - grayscale_img).astype('float64')
        G = np.sqrt(np.square(Ix) + np.square(Iy))
        eg = 1 - G / np.max(G)
        for i, t in enumerate(translator):
            fg[:, :, i] = imutils.translate(eg, *t)
            if i % 2 == 0:
                fg[:, :, i] = fg[:, :, i] / np.sqrt(2)

        Dp = np.zeros((*self.img.shape[:2], 2))
        Dp[:, :, 0] = np.where(G > 0, Iy / G, Iy)
        Dp[:, :, 1] = np.where(G > 0, -Ix / G, -Ix)
        dp = np.zeros((*self.img.shape[:2], 8))
        dq = np.zeros((*self.img.shape[:2], 8))
        for i, t in enumerate(translator):
            tile = np.tile(-np.array(t), (*self.img.shape[:2], 1)) / (np.sqrt(2) if i % 2 == 1 else 1)
            conds = np.zeros((*self.img.shape[:2], 2))
            conds[:, :, 0] = np.tensordot(Dp, -np.array(t), axes=(2, 0)) >= 0
            conds[:, :, 1] = np.tensordot(Dp, -np.array(t), axes=(2, 0)) >= 0
            L = np.where(conds, tile, -tile)
            dp[:, :, i] = np.sum(Dp * L, axis=2)
            dq[:, :, i] = np.sum(imutils.translate(Dp, *t) * L, axis=2)
        # There is a mistake in the paper, the max possible value of the sum of these 2 arccos is 1.5*pi, not pi.
        # So to make sure a max possible path cost of 1, we divide the thing by 1.5*pi instead.
        fd = 1/(1.5 * np.pi) * (np.arccos(dp) + np.arccos(dq))

        return wz * fz + wd * fd + wg * fg

    def compute_pixel_node(self):
        pixel_node = np.zeros((3 * self.h, 3 * self.w, 3))
        pixel_node[1::3, 1::3, :] = self.img
        return pixel_node.astype('uint8')

    def compute_cost_graph(self):
        """
        Generate the 'cost graph' of a given image.
        """
        if self.cost_matrix is None:
            self.cost_matrix = self.compute_cost()
        cost_graph = np.zeros((3 * self.h, 3 * self.w, 3))
        cost_graph[1::3, 2::3, :] = self.cost_matrix[:, :, 0:1]
        cost_graph[0::3, 2::3, :] = self.cost_matrix[:, :, 1:2]
        cost_graph[0::3, 1::3, :] = self.cost_matrix[:, :, 2:3]
        cost_graph[0::3, 0::3, :] = self.cost_matrix[:, :, 3:4]
        cost_graph[1::3, 0::3, :] = self.cost_matrix[:, :, 4:5]
        cost_graph[2::3, 0::3, :] = self.cost_matrix[:, :, 5:6]
        cost_graph[2::3, 1::3, :] = self.cost_matrix[:, :, 6:7]
        cost_graph[2::3, 2::3, :] = self.cost_matrix[:, :, 7:]
        cost_graph = cost_graph * (255 / np.max(cost_graph))  # normalize cost matrix for display purpose
        cost_graph[1::3, 1::3, :] = self.img  # add back the original image at pixel center
        return cost_graph.astype('uint8')

    def set_seed(self, x, y, use_snapping=None):
        """
        set seed with given point (x, y) in Qt coordinates, return the seed in Qt coordinates.
        """
        use_snapping = self.use_snapping if use_snapping is None else use_snapping
        if use_snapping:
            edge = self.nearest_edge(x, y)
            self.seed = (edge[1], edge[0])
        else:
            self.seed = (y, x)
        self.initialize_pixel_map()
        return self.seed[1], self.seed[0]

    def compute_pixel_map(self, dest=None, progress_callback=None):
        """
        Compute single-source shortest path from seed to each other vertices.
        """
        if self.cost_matrix is None:
            self.cost_matrix = self.compute_cost()
            self.initialize_pixel_map()
        if dest is not None:
            dest = tuple(reversed(dest))
        if not self.pq:
            heapq.heappush(self.pq, (self.pixel_map[self.seed].cost, self.pixel_map[self.seed]))
        while self.pq:
            cost, cur_node = heapq.heappop(self.pq)
            if cur_node.state == 2:
                continue
            cur_node.state = 2  # mark current node as expanded
            self.count += 1
            if progress_callback is not None:
                progress_callback.emit(self.count / (self.h * self.w))
            for ind, dr in enumerate(PixelNode.Directions):
                (x, y) = cur_node.pos + dr
                if not (0 <= x < self.h and 0 <= y < self.w):
                    continue
                next_node = self.pixel_map[x, y]
                if next_node.state != 2:
                    if next_node.state == 0:
                        next_node.prev = cur_node.pos
                        next_node.state = 1
                        next_node.cost = cur_node.cost + self.cost_matrix[cur_node.pos][ind]
                        next_node.level = cur_node.level + 1
                        if next_node.level > self.max_level:
                            self.max_level = next_node.level
                        heapq.heappush(self.pq, (next_node.cost, next_node))
                    elif cur_node.cost + self.cost_matrix[cur_node.pos][ind] < next_node.cost:
                        next_node.cost = cur_node.cost + self.cost_matrix[cur_node.pos][ind]
                        next_node.prev = cur_node.pos
                        next_node.level = cur_node.level + 1
                        if next_node.level > self.max_level:
                            self.max_level = next_node.level
                        heapq.heappush(self.pq, (next_node.cost, next_node))
            while self.pq and self.pq[0][1].state == 2:
                heapq.heappop(self.pq)
            if cur_node.pos == dest:
                break

    # def compute_pixel_map_fh(self):
    #     if self.cost_matrix is None:
    #         self.cost_matrix = self.compute_cost()
    #     self.initialize_pixel_map()
    #     self.items = {}
    #     pq = Fibonacci_heap()
    #     handle = pq.enqueue(self.pixel_map[self.seed], self.pixel_map[self.seed].cost)
    #     self.items[self.seed] = handle
    #     while pq:
    #         cur_node = pq.dequeue_min().m_elem
    #         cur_node.state = 2
    #         for ind, dr in enumerate(PixelNode.Directions):
    #             (x, y) = cur_node.pos + dr
    #             if not (0 <= x < self.img.shape[0] and 0 <= y < self.img.shape[1]):
    #                 continue
    #             next_node = self.pixel_map[x, y]
    #             if next_node.state != 2:
    #                 if next_node.state == 0:
    #                     next_node.prev = cur_node.pos
    #                     next_node.state = 1
    #                     next_node.cost = cur_node.cost + self.cost_matrix[cur_node.pos][ind]
    #                     handle = pq.enqueue(next_node, next_node.cost)
    #                     self.items[(x, y)] = handle
    #                 elif cur_node.cost + self.cost_matrix[cur_node.pos][ind] < next_node.cost:
    #                     next_node.cost = cur_node.cost + self.cost_matrix[cur_node.pos][ind]
    #                     next_node.prev = cur_node.pos
    #                     pq.decrease_key(self.items[next_node.pos], next_node.cost)

    def compute_path_tree(self, progress_callback=None):
        if self.count < self.w * self.h:
            self.compute_pixel_map(progress_callback=progress_callback)
        # tree_graph = np.zeros((self.h, self.w, 3))
        # for i in range(self.h):
        #     for j in range(self.w):
        #         tree_graph[i, j] = np.array([255, 255, 0]) * (self.pixel_map[i, j].level / self.max_level)  # yellow
        # return tree_graph.astype('uint8')
        tree_graph = np.zeros((3 * self.h, 3 * self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                if (i, j) != self.seed:
                    i_prev, j_prev = self.pixel_map[i, j].prev
                    edge = list(zip(*[(3 * i + t * (i_prev - i), 3 * j + t * (j_prev - j)) for t in range(3)]))
                    color_level = np.array([200, 200, 0]) * (self.pixel_map[i, j].level / self.max_level) \
                                  + np.array([55, 55, 0])  # set the lowest color level to 55 for avoiding complete darkness
                    tree_graph[edge] = color_level
        tree_graph[1::3, 1::3, :] = self.img
        return tree_graph.astype('uint8')

    def get_path(self, x, y):
        if self.pixel_map is None or self.pixel_map[y, x].state != 2:
            self.compute_pixel_map(dest=(x, y))
        path = deque()
        while (y, x) != self.seed:
            path.appendleft((x, y))
            y, x = self.pixel_map[(y, x)].prev
        path.appendleft((x, y))
        return list(path)

    def commit_path(self, x, y):
        if self.use_snapping:
            x, y = self.nearest_edge(x, y)
        path = self.get_path(x, y)
        self.paths.append(path)
        self.set_seed(x, y)

    def pop_path(self):
        removed_path = self.paths.pop()
        seed = removed_path[0]
        self.set_seed(seed[0], seed[1], use_snapping=False)

    def save_contour(self):
        first_seed = self.paths[0][0]
        self.compute_pixel_map(dest=self.paths[0][0])
        last_path = self.get_path(*first_seed)
        self.paths.append(last_path)
        contour = list(chain(*self.paths))
        self.contours.append(contour)
        self.paths = []
        self.seed = None

    def pop_contour(self):
        self.contours.pop()

    def create_mask(self):
        if not self.contours:
            raise Exception("Please save a contour before creating a mask.")
        visited = set(chain(*self.contours))
        mask = np.full(self.img.shape[:2], True)
        # start flood fill from a corner (outside the mask) until we hit the contour
        corners = [(0, 0), (0, self.h - 1), (self.w - 1, 0), (self.w - 1, self.h - 1)]
        start_point = None
        for corner in corners:
            if corner not in visited:
                start_point = corner
                break
        if start_point is None:
            raise Exception("Error creating mask. Please check your contour.")
        stack = [start_point]
        while stack:
            y, x = stack.pop()
            mask[x, y] = False
            visited.add((y, x))
            neighbors = [(y, x-1), (y, x+1), (y-1, x), (y+1, x)]
            for yn, xn in neighbors:
                if not (0 <= xn < self.h and 0 <= yn < self.w):
                    continue
                if (yn, xn) not in visited:
                    stack.append((yn, xn))
        return mask

    def nearest_edge(self, y_cor, x_cor, limit=10):
        """
        Find the nearest edge from the current pos (y_cor, x_cor) in Qt coordinates,
        return the point also in Qt coordinates.
        """
        start = (0, y_cor, x_cor)
        visited = set()
        q = deque()
        q.append(start)
        while q:
            dist, y, x = q.popleft()
            if limit is not None and dist > limit:
                break
            visited.add((y, x))
            if self.edges[x, y] > 0:
                return y, x
            neighbors = [(y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)]
            for yn, xn in neighbors:
                if not (0 <= xn < self.h and 0 <= yn < self.w):
                    continue
                if (yn, xn) not in visited:
                    q.append((dist + 1, yn, xn))
        return y_cor, x_cor


if __name__ == "__main__":
    img = plt.imread("sample/ferry.bmp")
    iscr = IScissor()
    iscr.set_image(img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cost_matrix = iscr.compute_cost()
    # cost_graph = iscr.compute_cost_graph()
    # plt.imshow(cost_graph)
    # plt.show()

    # cv2.imshow('image', cost_graph.astype('uint8'))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # iscr.use_snapping = True
    iscr.set_seed(115, 85)
    # iscr.compute_pixel_map()
    # iscr.get_path(63, 27)
    # iscr.get_path(20, 10)
    iscr.commit_path(211, 141)
    iscr.commit_path(107, 145)
    # iscr.pop_path()
    # iscr.commit_path(152, 61)
    iscr.save_contour()

    path_img = np.zeros(iscr.img.shape, dtype='uint8')
    for y, x in iscr.contours[0]:
        path_img[x, y, 0] = 255

    overlay = cv2.addWeighted(iscr.img, 0.5, path_img, 1, 0)
    plt.imshow(overlay)
    plt.show()

    mask = iscr.create_mask()
    plt.imshow(mask, cmap="gray")
    plt.show()

    # cv2.imshow("path_img", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # costmap = np.array([[iscr.pixel_map[i][j].cost for j in range(h)] for i in range(w)])
    # costmap = costmap / np.max(costmap)
    # cv2.imshow('costmap', costmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # edge detection
    # edges = imutils.auto_canny(iscr.img)
    # plt.imshow(edges,cmap = 'gray')

