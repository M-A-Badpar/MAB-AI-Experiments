"""
Adversarial Game Tree Search: Minimax and Alpha-Beta Pruning.
Author: Mohammad Amin Badpar
"""

from typing import Any, List, Optional, Tuple


def minimax(position: Any, depth: int, max_player: bool) -> Tuple[float, Optional[Any]]:
    """Standard Minimax recursive search algorithm."""
    if depth == 0 or position.winner() is not None:
        return position.evaluate(), position

    if max_player:
        best_point = float("-inf")
        best_move = None
        for move in get_all_moves(position, WHITE):
            current_point, _ = minimax(move, depth - 1, False)
            if current_point > best_point:
                best_point = current_point
                best_move = move
        return best_point, best_move

    else:
        min_point = float("inf")
        best_move = None
        for move in get_all_moves(position, RED):
            current_point, _ = minimax(move, depth - 1, True)
            if current_point < min_point:
                min_point = current_point
                best_move = move
        return min_point, best_move


def minimax_alpha_beta(position: Any, depth: int, alpha: float, beta: float, max_player: bool) -> Tuple[float, Optional[Any]]:
    """Minimax search optimized with Alpha-Beta pruning."""
    if depth == 0 or position.winner() is not None:
        return position.evaluate(), position

    if max_player:
        best_point = float("-inf")
        best_move = None
        for move in get_all_moves(position, WHITE):
            current_point, _ = minimax_alpha_beta(move, depth - 1, alpha, beta, False)
            if current_point > best_point:
                best_point = current_point
                best_move = move

            alpha = max(alpha, current_point)
            if beta <= alpha:
                break  # Beta cutoff

        return best_point, best_move

    else:
        min_point = float("inf")
        best_move = None
        for move in get_all_moves(position, RED):
            current_point, _ = minimax_alpha_beta(move, depth - 1, alpha, beta, True)
            if current_point < min_point:
                min_point = current_point
                best_move = move

            beta = min(beta, current_point)
            if beta <= alpha:
                break  # Alpha cutoff

        return min_point, best_move


if __name__ == "__main__":
    print("Adversarial Search Engine Initialized")
    print("Available Algorithms: [minimax, minimax_alpha_beta]")
