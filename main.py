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

def is_putable_position_num(board, is_black, position_num):
    position_nums = get_flip_positions(board, is_black, position_num)
    return len(position_nums) != 0

def get_putable_position_nums(board, is_black):
    return [num for num in range(64) if is_putable_position_num(board, is_black, num)]

def get_player(board, is_black = True):
    return (board, is_black, get_putable_position_nums(board, is_black))

def is_end_board(board):
    return len(np.where(board == 0)[0]) == 0

def put(player, position_num):
    board, is_black, _ = player
    board = board.copy()
    own = 1 if is_black else -1
    position_nums = get_flip_positions(board, is_black, position_num)
    if len(position_nums) > 0:
        board[position_num] = own
        for position_num in position_nums:
            board[position_num] = own
    return board

def is_putable(player):
    _, _, putable_position_nums = player
    return len(putable_position_nums) > 0

def render_board(player):
    board, is_black, putable_position_nums = player
    black, white = "○", "●"  # 1, -1
    display_board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board)]
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*display_board))
    print("{}: putable position numbers are {}".format(black if is_black else white, putable_position_nums))

def is_pass_last_put(game):
    if len(game) == 0:
        return False
    _, position_num = game[-1]
    return position_num == None

def is_end_game(game, player):
    board, _, _ = player
    return is_end_board(board) or (is_pass_last_put(game) and not is_putable(player))

def choice_random(putable_position_nums):
    return np.random.choice(putable_position_nums)

def choice_human(putable_position_nums):
    choice = None
    while True:
        try:
            choice = input("Please enter number in {}: ".format(putable_position_nums))
            choice = int(choice)
            if choice in putable_position_nums:
                break
            else:
                print("{} is invalid".format(choice))
        except Exception:
            print("{} is invalid".format(choice))
    return choice

def main():
    game = []
    player = get_player(get_init_board())
    while True:
        board, is_black, putable_position_nums = player
        render_board(player)
        if is_end_game(game, player):
            break
        position_num = None
        if is_putable(player):
            if is_black:
                position_num = choice_human(putable_position_nums)
            else:
                position_num = choice_random(putable_position_nums)
            print(position_num)
            board = put(player, position_num)
        else:
            print("pass")
        game.append((player, position_num))
        player = get_player(board, not is_black)

if __name__ == "__main__":
    main()
