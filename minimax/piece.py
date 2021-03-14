from minimax.constants import SQUARE_SIZE
import pygame


"""
Class representing Piece on Board

:param row: row of Piece
:type row: int
:param col: col of Piece
:type col: int
"""
class Piece():
    PADDING = 10
    OUTLINE  = 2

    def __init__(self, row, col, color) -> None:
        self.row = row
        self.col = col
        self.color = color

        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def draw(self, window):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(window, self.color, (self.x, self.y), radius + self.OUTLINE)
        pygame.draw.circle(window, self.color, (self.x, self.y), radius)
        