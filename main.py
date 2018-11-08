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

def get_stone_num(board):
    black_num = len(np.where(board ==  1)[0])
    white_num = len(np.where(board == -1)[0])
    return black_num, white_num

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

def playout(player, position_num):
    is_win = False
    _, is_black, _ = player
    board = put(player, position_num)
    puts = game(choice_random, choice_random, board, False)
    if len(puts) > 0:
        player_last, _ = puts[-1]
        board_last, _, _ = player_last
        black_num, white_num = get_stone_num(board_last)
        if (is_black and black_num > white_num) or (not is_black and black_num < white_num):
            is_win = True
    return is_win

def is_putable(player):
    _, _, putable_position_nums = player
    return len(putable_position_nums) > 0

def render_board(player):
    board, is_black, putable_position_nums = player
    black, white = "○", "●"  # 1, -1
    black_num, white_num = get_stone_num(board)
    display_board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board)]
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*display_board))
    print("{}:{} {}:{}".format(black, black_num, white, white_num))
    print("{}: putable position numbers are {}".format(black if is_black else white, putable_position_nums))

def is_pass_last_put(game):
    if len(game) == 0:
        return False
    _, position_num = game[-1]
    return position_num == None

def is_end_game(game, player):
    board, _, _ = player
    return is_end_board(board) or (is_pass_last_put(game) and not is_putable(player))

def choice_random(player):
    _, _, putable_position_nums = player
    return np.random.choice(putable_position_nums)

def choice_primitive_monte_carlo(player, try_num = 150):
    _, _, putable_position_nums = player
    position_scores = np.zeros(len(putable_position_nums))
    for _ in range(try_num):
        playouts = [playout(player, position_num) for position_num in putable_position_nums]
        position_scores += np.array([1 if is_win else 0 for is_win in playouts])
    index = np.random.choice(np.where(position_scores == position_scores.max())[0])
    return putable_position_nums[index]

def choice_human(player):
    _, _, putable_position_nums = player
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

def game(choice_black, choice_white, board = None, is_render = True):
    game = []
    if board is None:
        board = get_init_board()
    player = get_player(board)
    while True:
        board, is_black, _ = player
        if is_render:
            render_board(player)
        if is_end_game(game, player):
            break
        position_num = None
        if is_putable(player):
            choice = choice_black if is_black else choice_white
            position_num = choice(player)
            if is_render:
                print(position_num)
            board = put(player, position_num)
        else:
            if is_render:
                print("pass")
        game.append((player, position_num))
        player = get_player(board, not is_black)
    return game

def main():
    game(choice_human, choice_primitive_monte_carlo)

if __name__ == "__main__":
    main()
