from minimax.game_tree import Game_tree
from minimax.minimax import minimax
from minimax.alpha_beta import alpha_beta
import pygame
from minimax.constants import WIDTH, HEIGHT
from minimax.board import Board
import time

def main():
    run_auto("minimax", 3, True, [7, 0], True)

def run_auto(algorithm, depth, max_move, wolf_pos, randomly):
    
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Wolf and Sheep')

    start = time.time()
    game_tree = Game_tree(wolf_pos)

    current_state = game_tree.root
    current_move = max_move

    while current_state is not None:

        board = Board(current_state.matrix)
        board.draw(window)
        pygame.display.update()
        # Displays every move at the board
        pygame.time.delay(1000)

        if algorithm == "minimax":
            minimax(current_state, depth, current_move, randomly)
        elif algorithm == "alpha-beta":
            alpha_beta(current_state, depth, current_move, float('-inf'), float('inf'))
        
        current_move = not current_move
        current_state = current_state.best_child

    end = time.time()
    pygame.quit()
    print(f"TIME: {end-start}\n")

def run_click(algorithm, depth, max_move, wolf_pos, randomly):
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Wolf and Sheep')
    
    game_tree = Game_tree(wolf_pos)

    current_state = game_tree.root
    current_move = max_move

    while current_state is not None:

        board = Board(current_state.matrix)
        board.draw(window)
        pygame.display.update()

        if algorithm == "minimax":
            minimax(current_state, depth, current_move, randomly)
        elif algorithm == "alpha-beta":
            alpha_beta(current_state, depth, current_move, float('-inf'), float('inf'))

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                current_state = current_state.best_child
                current_move = not current_move
            elif event.type == pygame.QUIT:
                pygame.quit()

    pygame.quit()

if __name__ == "__main__":
    main()