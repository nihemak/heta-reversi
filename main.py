# coding: utf-8

import numpy as np

def render_board(board_data):
    black = "○"  # 1
    white = "●"  # -1
    board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board_data)]
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*board))

def main():
    board = np.array([0] * 64, dtype=np.float32)
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    render_board(board)

if __name__ == "__main__":
    main()
