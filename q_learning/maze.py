import numpy as np

class Maze():

    def __init__(self, n, start, end, matrix=None):
        self.n = n if n > 0 else 8
        self.start = start if self.valid_coord(start[0], start[1]) else (0, n-1)
        self.end = end if self.valid_coord(end[0], end[1]) else (n-1, 0)
        if matrix is None:
            self.generate()
        else:
            self.matrix = matrix

    def valid_coord(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n

    def generate(self):
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            row = np.random.choice(2, size=self.n)
            matrix[i] += row

        matrix[self.start[0], self.start[1]] = 0
        matrix[self.end[0], self.end[1]] = 0

        self.matrix = matrix
        visited = list()
        discovered = set()
        visited.append(self.start)
        while len(visited):
            v = visited.pop(len(visited) - 1)
            if (v not in discovered):
                discovered.add(v)
                if (self.in_matrix(v[0], v[1] - 1)):
                    if not self.is_hole(v[0], v[1] - 1):
                        visited.append((v[0], v[1] - 1))
                if (self.in_matrix(v[0], v[1] + 1)):
                    if not self.is_hole(v[0], v[1] + 1):
                        visited.append((v[0], v[1] + 1))
                if (self.in_matrix(v[0] - 1, v[1])):
                    if not self.is_hole(v[0] - 1, v[1]):
                        visited.append((v[0] - 1, v[1]))
                if (self.in_matrix(v[0] + 1, v[1])):
                    if not self.is_hole(v[0] + 1, v[1]):
                        visited.append((v[0] + 1, v[1]))

        if self.start not in discovered or self.end not in discovered:
            self.generate()


    def in_matrix(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n

    def is_hole(self, x, y):
        return self.matrix[x, y] == 1

    def get_matrix(self):
        return self.matrix

    def get_n(self):
        return self.n

    def is_terminal(self, x, y):
        return self.matrix[x, y] == 1 or (x, y) == self.end
