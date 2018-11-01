# coding: utf-8

import numpy as np

def is_putable_position_row_column(board, row, column, row_add, column_add):
    row    += row_add
    column += column_add
    exists = False
    while row >= 0 and row < 8 and column >= 0 and column < 8:
        if exists == True and board[row * 8 + column] == 1:
            return True
        elif board[row * 8 + column] != -1:
            return False

        exists = True
        row    += row_add
        column += column_add

    return False

def is_putable_position_num(board, position_num):
    if not (position_num >= 0 and position_num <= 63):
        return False
    if board[position_num] != 0:
        return False

    column = position_num % 8
    row    = int((position_num - column) / 8)

    row_column_adds = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
    for row_add, column_add in row_column_adds:
        if is_putable_position_row_column(board, row, column, row_add, column_add):
            return True

    return False

def get_putable_position_nums(board):
    return [num for num in range(64) if is_putable_position_num(board, num)]

def render_board(board_data):
    black = "○"  # 1
    white = "●"  # -1
    board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board_data)]
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*board))
    print("{}: putable position numbers are {}".format(black, get_putable_position_nums(board_data)))

def main():
    board = np.array([0] * 64, dtype=np.float32)
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    render_board(board)

if __name__ == "__main__":
    main()
