import pygame
from minimax.constants import BLACK, ROWS, RED, SQUARE_SIZE, COLS, WHITE
from minimax.piece import Piece

"""
Class representing Board of Wolf and Sheep game

:param matrix: matrix of state
:type matrix: list
:param board: matrix of state with Pieces included
:type board: list
"""

class Board():

    def __init__(self, matrix) -> None:
        self.matrix = matrix
        self.board = []
        self.create_board()
        
    def draw_squares(self, window):
        window.fill(BLACK)
        for row in range(len(self.matrix)):
            for col in range(row % 2, len(self.matrix[0]), 2):
                pygame.draw.rect(window, RED, (row*SQUARE_SIZE, col*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def create_board(self):
        for row in range(len(self.matrix)):
            self.board.append([])
            for col in range(len(self.matrix[0])):
                if self.matrix[row][col] == 1:
                    self.board[row].append(Piece(row, col, WHITE))
                elif self.matrix[row][col] == -1:
                    self.board[row].append(Piece(row, col, RED))
                else:
                    self.board[row].append(0)

    def draw(self, window):
        self.draw_squares(window)
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[0])):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(window)