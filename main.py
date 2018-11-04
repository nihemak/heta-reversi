# coding: utf-8

import numpy as np

def get_init_board():
    board = np.array([0] * 64, dtype=np.float32)
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    return board

def get_flip_positions_by_row_column(board, is_black, row, column, row_add, column_add):
    position_nums = []

    own, pair = (1, -1) if is_black else (-1, 1)

    row    += row_add
    column += column_add
    exists = False
    while row >= 0 and row < 8 and column >= 0 and column < 8:
        position_num = row * 8 + column

        if exists == True and board[position_num] == own:
            break
        if board[position_num] != pair:
            position_nums = []
            break

        position_nums.append(position_num)

        exists = True
        row    += row_add
        column += column_add

    return position_nums

def get_flip_positions(board, is_black, position_num):
    position_nums = []

    if not (position_num >= 0 and position_num <= 63):
        return position_nums
    if board[position_num] != 0:
        return position_nums

    column = position_num % 8
    row    = int((position_num - column) / 8)

    row_column_adds = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
    for row_add, column_add in row_column_adds:
        position_nums.extend(get_flip_positions_by_row_column(board, is_black, row, column, row_add, column_add))

    return position_nums

def put(board, is_black, position_num):
    own = 1 if is_black else -1
    position_nums = get_flip_positions(board, is_black, position_num)
    if len(position_nums) > 0:
        board[position_num] = own
        for position_num in position_nums:
            board[position_num] = own
    return board

def is_putable_position_num(board, is_black, position_num):
    position_nums = get_flip_positions(board, is_black, position_num)
    return len(position_nums) != 0

def get_putable_position_nums(board, is_black):
    return [num for num in range(64) if is_putable_position_num(board, is_black, num)]

def is_putable(board, is_black):
    return len(get_putable_position_nums(board, is_black)) > 0

def render_board(board, is_black):
    black, white = "○", "●"  # 1, -1
    display_board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board)]
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*display_board))
    print("{}: putable position numbers are {}".format(black if is_black else white, get_putable_position_nums(board, is_black)))

def is_end(board):
    return not (is_putable(board, True) or is_putable(board, False))

def main():
    board = get_init_board()

    is_black = True
    while not is_end(board):
        render_board(board, is_black)
        if is_putable(board, is_black):
            position_num = np.random.choice(get_putable_position_nums(board, is_black))
            print(position_num)
            board = put(board, is_black, position_num)
        else:
            print("pass")
        is_black = not is_black
    render_board(board, is_black)

if __name__ == "__main__":
    main()
