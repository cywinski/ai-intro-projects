import numpy as np
import pandas as pd

"""
Class representing particular state of Wolf and Sheep game

:param wolf_pos: position of wolf
:type wolf_pos: list
:param sheep_pos: position of sheep
:type sheep_pos: list
:param parent: parent of state
:type parent: State
:param best_child: successor with best evaluation for either Max or Min player
:type best_child: State
:param matrix: matrix of state
:type matrix: list
:param game_over: is game over yet
:type game_over: boolean
:param successors: all successors of state
:type successors: list
"""

class State():

    def __init__(self, wolf_pos, sheep_pos, parent):
        self.wolf_pos = wolf_pos
        self.sheep_pos = sheep_pos
        self.parent = parent
        self.best_child = None
        self.matrix = self.gen_matrix()
        self.game_over = False
        self.successors = []

    def gen_matrix(self):
        chb_matrix = np.zeros((8, 8), dtype=int)
        chb_matrix[self.wolf_pos[0]][self.wolf_pos[1]] = 1
        for pos in self.sheep_pos:
            chb_matrix[pos[0]][pos[1]] = -1

        return chb_matrix

    def is_game_over(self):
        return self.game_over

    def __str__(self):
        column_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        row_names = [1, 2, 3, 4, 5, 6, 7, 8]
        df = pd.DataFrame(self.matrix, columns=column_names, index=row_names)
        return str(df)

    def is_terminal(self):
        # Wolf reached one of the end points
        if self.wolf_pos[0] == 0:
            return True

        # Sheep blocked wolf
        # Wolf trapped at the left or right edge of chessboard
        if self.wolf_pos[1] in {0, 7} and [self.wolf_pos[0] - 1, self.wolf_pos[1] + 1] in self.sheep_pos \
            and [self.wolf_pos[0] + 1, self.wolf_pos[1] + 1] in self.sheep_pos:
            self.game_over = True
            return True

        # Wolf trapped in left-down corner of chessboard
        if self.wolf_pos == [7, 0] and [6, 1] in self.sheep_pos:
            self.game_over = True
            return True

        return False

    def evaluate(self):
        return self.evaluate_wolf() - self.evaluate_sheep()

    def evaluate_wolf(self):
        behind = True
        empty_cols = True
        # As default hauristics for wolf will be:
        # 8 - row_of_wolf_position + number of sheep behind wolf
        result = 16 - 2*self.wolf_pos[0]

        for sheep in self.sheep_pos:
            if sheep[0] <= self.wolf_pos[0]:
                behind = False

        if behind:
            result += 100

        return result

    def evaluate_sheep(self):
        result = 0
        min_row = float('inf')
        max_row = float('-inf')

        for pos in self.sheep_pos:
            result += 8 - max(abs(self.wolf_pos[0] - pos[0]), abs(self.wolf_pos[1] - pos[1]))

            # Wolf is on the edge of the board
            if self.wolf_pos[0] == 0 or self.wolf_pos[1] in {0, 1}:
                if max(abs(self.wolf_pos[0] - pos[0]), abs(self.wolf_pos[1] - pos[1])) == 1:
                    result += 1

            if pos[0] < min_row:
                min_row = pos[0]
            if pos[0] > max_row:
                max_row = pos[0]

        if max_row - min_row < 2:
            result += 2            

        if self.is_state_valid(self.wolf_pos[0] - 1, self.wolf_pos[1] - 1):
            if self.matrix[self.wolf_pos[0] - 1][self.wolf_pos[1] - 1] == -1:
                result += 1

        if self.is_state_valid(self.wolf_pos[0] - 1, self.wolf_pos[1] + 1):
            if self.matrix[self.wolf_pos[0] - 1][self.wolf_pos[1] + 1] == -1:
                result += 1

        return result / 4

    def get_wolf_pos(self):
        return self.wolf_pos

    def get_sheep_pos(self):
        return self.sheep_pos

    def get_matrix(self):
        return self.matrix

    def get_successors(self):
        return self.successors

    def is_state_valid(self, row, col):
        # Check if range is not out of chessboard
        if row >= 0 and row <= 7 and col >= 0 and col <= 7:
            # Check if position is empty
            if self.get_matrix()[row][col] == 0:
                return True

        return False

    def make_successors(self, max_move):
        if max_move:
            possible_moves = [[self.wolf_pos[0] - 1, self.wolf_pos[1] + 1],
                              [self.wolf_pos[0] - 1, self.wolf_pos[1] - 1],
                              [self.wolf_pos[0] + 1, self.wolf_pos[1] + 1],
                              [self.wolf_pos[0] + 1, self.wolf_pos[1] - 1]]
            for move in possible_moves:
                if self.is_state_valid(move[0], move[1]):
                    self.successors.append(State(move, self.sheep_pos, self))

        else:
            for i in range(len(self.sheep_pos)):
                possible_moves = [[self.sheep_pos[i][0] + 1, self.sheep_pos[i][1] - 1],
                                  [self.sheep_pos[i][0] + 1, self.sheep_pos[i][1] + 1]]
                for move in possible_moves:
                    if self.is_state_valid(move[0], move[1]):
                        new_sheep = self.sheep_pos.copy()
                        # Change position of particular sheep
                        new_sheep[i] = move
                        self.successors.append(State(self.wolf_pos, new_sheep, self))

    def set_best_child(self, state):
        self.best_child = state

    def set_parent(self, state):
        self.parent = state
