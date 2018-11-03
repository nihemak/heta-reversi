# coding: utf-8

import numpy as np

def get_flip_positions_by_row_column(board, row, column, row_add, column_add):
    row    += row_add
    column += column_add
    exists = False
    position_nums = []
    while row >= 0 and row < 8 and column >= 0 and column < 8:
        position_num = row * 8 + column

        if exists == True and board[position_num] == 1:
            break
        elif board[position_num] != -1:
            position_nums = []
            break

        position_nums.append(position_num)

        exists = True
        row    += row_add
        column += column_add

    return position_nums

def put_black(board, position_num):
    column = position_num % 8
    row    = int((position_num - column) / 8)

    row_column_adds = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))

    board[position_num] = 1
    for row_add, column_add in row_column_adds:
        position_nums = get_flip_positions_by_row_column(board, row, column, row_add, column_add)
        for position_num in position_nums:
            board[position_num] = 1

    return board

def is_putable_position_num(board, position_num):
    if not (position_num >= 0 and position_num <= 63):
        return False
    if board[position_num] != 0:
        return False

    column = position_num % 8
    row    = int((position_num - column) / 8)

    row_column_adds = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
    for row_add, column_add in row_column_adds:
        position_nums = get_flip_positions_by_row_column(board, row, column, row_add, column_add)
        if len(position_nums) != 0:
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
    position_num = np.random.choice(get_putable_position_nums(board))
    print(position_num)
    board = put_black(board, position_num)
    render_board(board)

if __name__ == "__main__":
    main()
