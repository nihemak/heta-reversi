# coding: utf-8

import numpy as np

def choice_random(player):
    _, _, putable_position_nums = player
    choice_data = { 'position_num': np.random.choice(putable_position_nums) }
    return choice_data

class Reversi:
    @classmethod
    def get_init_board(cls):
        board = np.array([0] * 64, dtype=np.float32)
        board[28] = board[35] = 1
        board[27] = board[36] = -1
        return board

    @classmethod
    def get_flip_positions_by_row_column(cls, board, is_black, row, column, row_add, column_add):
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

    @classmethod
    def get_flip_positions(cls, board, is_black, position_num):
        position_nums = []

        if not (position_num >= 0 and position_num <= 63):
            return position_nums
        if board[position_num] != 0:
            return position_nums

        column = position_num % 8
        row    = int((position_num - column) / 8)

        row_column_adds = ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
        for row_add, column_add in row_column_adds:
            position_nums.extend(cls.get_flip_positions_by_row_column(board, is_black, row, column, row_add, column_add))

        return position_nums

    @classmethod
    def is_putable_position_num(cls, board, is_black, position_num):
        position_nums = cls.get_flip_positions(board, is_black, position_num)
        return len(position_nums) != 0

    @classmethod
    def get_putable_position_nums(cls, board, is_black):
        return [num for num in range(64) if cls.is_putable_position_num(board, is_black, num)]

    @classmethod
    def get_player(cls, board, is_black = True):
        return (board, is_black, cls.get_putable_position_nums(board, is_black))

    @classmethod
    def is_end_board(cls, board):
        return len(np.where(board == 0)[0]) == 0

    @classmethod
    def get_stone_num(cls, board):
        black_num = len(np.where(board ==  1)[0])
        white_num = len(np.where(board == -1)[0])
        return black_num, white_num

    @classmethod
    def put(cls, player, position_num):
        board, is_black, _ = player
        board = board.copy()
        if position_num is not None:
            own = 1 if is_black else -1
            position_nums = cls.get_flip_positions(board, is_black, position_num)
            if len(position_nums) > 0:
                board[position_num] = own
                for position_num in position_nums:
                    board[position_num] = own
        return board

    @classmethod
    def playout(cls, player, position_num):
        _, is_black, _ = player
        board = cls.put(player, position_num)
        puts = cls.game(choice_random, choice_random, board, False)
        is_win = cls.is_win_game(puts, is_black)
        return is_win

    @classmethod
    def is_putable(cls, player):
        _, _, putable_position_nums = player
        return len(putable_position_nums) > 0

    @classmethod
    def render_board(cls, player):
        board, is_black, putable_position_nums = player
        black, white = "○", "●"  # 1, -1
        black_num, white_num = cls.get_stone_num(board)
        display_board = [i if v == 0 else " {} ".format(black if v == 1 else white) for i, v in enumerate(board)]
        row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
        hr = "\n------------------------------------------------\n"
        layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
        print((layout).format(*display_board))
        print("{}:{} {}:{}".format(black, black_num, white, white_num))
        print("{}: putable position numbers are {}".format(black if is_black else white, putable_position_nums))

    @classmethod
    def is_pass_last_put(cls, game):
        if len(game) == 0:
            return False
        _, choice_data = game[-1]
        position_num = choice_data['position_num']
        return position_num == None

    @classmethod
    def is_win_game(cls, game, is_black):
        is_win = False
        if len(game) > 0:
            player_last, _ = game[-1]
            board_last, _, _ = player_last
            black_num, white_num = cls.get_stone_num(board_last)
            if (is_black and black_num > white_num) or (not is_black and black_num < white_num):
                is_win = True
        return is_win

    @classmethod
    def is_end_game(cls, game, player):
        board, _, _ = player
        return cls.is_end_board(board) or (cls.is_pass_last_put(game) and not cls.is_putable(player)) 

    @classmethod
    def game(cls, choice_black, choice_white, board = None, is_render = True, limit_step_num = None):
        steps = []
        if board is None:
            board = cls.get_init_board()
        player = cls.get_player(board)
        step_num = 0
        while True:
            if limit_step_num is not None and step_num >= limit_step_num:
                break
            board, is_black, _ = player
            if is_render:
                cls.render_board(player)
            if cls.is_end_game(steps, player):
                break
            position_num = None
            if cls.is_putable(player):
                choice = choice_black if is_black else choice_white
                choice_data = choice(player)
                position_num = choice_data["position_num"]
                if is_render:
                    print(position_num)
                board = cls.put(player, position_num)
            else:
                if is_render:
                    print("pass")
            steps.append((player, choice_data))
            player = cls.get_player(board, not is_black)
            step_num += 1
        return steps
