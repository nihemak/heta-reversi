# coding: utf-8

import numpy as np
import math
import functools
import json
import datetime
import sys
import chainer
import chainer.functions as F
import chainer.links as L

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
    if position_num is not None:
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
    _, choice_data = game[-1]
    position_num = choice_data['position_num']
    return position_num == None

def is_end_game(game, player):
    board, _, _ = player
    return is_end_board(board) or (is_pass_last_put(game) and not is_putable(player))

class DualNet(chainer.Chain):
    def __init__(self):
        super(DualNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(4,  48, 3, pad=1)
            self.conv1 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv2 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv3 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv4 = L.Convolution2D(48, 48, 3, pad=1)

            self.bn0 = L.BatchNormalization(48)
            self.bn1 = L.BatchNormalization(48)
            self.bn2 = L.BatchNormalization(48)
            self.bn3 = L.BatchNormalization(48)
            self.bn4 = L.BatchNormalization(48)

            self.conv_p1 = L.Convolution2D(48, 2, 1)
            self.bn_p1   = L.BatchNormalization(2)
            self.fc_p2   = L.Linear(8 * 8 * 2, 8 * 8)

            self.conv_v1 = L.Convolution2D(48, 1, 1)
            self.bn_v1   = L.BatchNormalization(1)
            self.fc_v2   = L.Linear(8 * 8, 48)
            self.fc_v3   = L.Linear(48, 1)

    def __call__(self, x):
        # tiny ResNet
        h0 = F.relu(self.bn0(self.conv0(x)))
        h1 = F.relu(self.bn1(self.conv1(h0)))
        h2 = F.relu(self.bn2(self.conv2(h1)) + h0)
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)) + h2)

        h_p1 = F.relu(self.bn_p1(self.conv_p1(h4)))
        policy = self.fc_p2(h_p1)

        h_v1  = F.relu(self.bn_v1(self.conv_v1(h4)))
        h_v2  = F.relu(self.fc_v2(h_v1))
        value = F.tanh(self.fc_v3(h_v2))

        return policy, value

class ChoiceReplaySteps:
    def __init__(self, steps):
        self._i = 0
        self._steps = steps

    def __call__(self, player):
        _, _, putable_position_nums = player
        # skip -1
        while True:
            step = self._steps[self._i]
            if step != -1:
                break
            self._i += 1
        if step not in putable_position_nums:
            step = np.random.choice(putable_position_nums)
        self._i += 1
        choice_data = { 'position_num': step }
        return choice_data

def choice_random(player):
    _, _, putable_position_nums = player
    choice_data = { 'position_num': np.random.choice(putable_position_nums) }
    return choice_data

def choice_primitive_monte_carlo(player, try_num = 150):
    _, _, putable_position_nums = player
    position_scores = np.zeros(len(putable_position_nums))
    for _ in range(try_num):
        playouts = [playout(player, position_num) for position_num in putable_position_nums]
        position_scores += np.array([1 if is_win else 0 for is_win in playouts])
    index = np.random.choice(np.where(position_scores == position_scores.max())[0])
    choice_data = { 'position_num': putable_position_nums[index] }
    return choice_data

class ChoiceMonteCarloTreeSearch:
    def _get_node(self, player, position_num):
        return {
            'player': player,
            'position_num': position_num,
            'try_num': 0,
            'win_num': 0,
            'child_nodes': None
        }

    def _get_initial_nodes(self, player):
        board, is_black, putable_position_nums = player
        nodes = [self._get_node(get_player(put(player, position_num), not is_black), position_num) for position_num in putable_position_nums]
        if len(putable_position_nums) == 0:
            nodes = [self._get_node(get_player(board, not is_black), None)]
        return nodes

    def _get_ucb1(self, node, total_num):
        return (node['win_num'] / node['try_num']) + math.sqrt(2 * math.log(total_num) / node['try_num'])

    def _selection_node_index(self, nodes):
        total_num = functools.reduce(lambda total_num, node: total_num + node['try_num'], nodes, 0)
        ucb1s = np.array([self._get_ucb1(node, total_num) if node['try_num'] != 0 else -1 for node in nodes])
        indexs = np.where(ucb1s == -1)[0]  # -1 is infinity
        if len(indexs) == 0:
            indexs = np.where(ucb1s == ucb1s.max())[0]
        index = np.random.choice(indexs)
        return index

    def _selection_expansion(self, nodes, expansion_num):
        game = []
        node, path = None, []
        target_nodes = nodes
        while True:
            index = self._selection_node_index(target_nodes)
            path.append(index)
            node = target_nodes[index]
            if node['child_nodes'] is None:
                if node['try_num'] >= expansion_num:
                    if is_end_game(game, node['player']):
                        break
                    # expansion
                    node['child_nodes'] = self._get_initial_nodes(node['player'])
                else:
                    break
            target_nodes = node['child_nodes']
            choice_data = { 'position_num': node['position_num'] }
            game.append((node['player'], choice_data))
        return nodes, node, path

    def _evaluation(self, node, is_black):
        is_win = playout(node['player'], node['position_num'])
        _, node_is_black, _ = node['player']
        node_is_win = (is_black == node_is_black and is_win) or (is_black != node_is_black and not is_win)
        return node_is_win

    def _backup(self, nodes, path, is_win):
        target_nodes = nodes
        for index in path:
            target_nodes[index]['try_num'] += 1
            if is_win:
                target_nodes[index]['win_num'] += 1
            target_nodes = target_nodes[index]['child_nodes']
        return nodes

    def _choice_node_index(self, nodes):
        try_nums = np.array([node['try_num'] for node in nodes])
        indexs = np.where(try_nums == try_nums.max())[0]
        index = np.random.choice(indexs)
        return index

    def __call__(self, player, try_num = 1500, expansion_num = 5):
        _, is_black, _ = player
        nodes = self._get_initial_nodes(player)
        for _ in range(try_num):
            nodes, node, path = self._selection_expansion(nodes, expansion_num)
            is_win = self._evaluation(node, is_black)
            nodes = self._backup(nodes, path, is_win)
        index = self._choice_node_index(nodes)
        choice = nodes[index]['position_num']
        choice_data = { 'position_num': choice }
        return choice_data

class ChoiceSupervisedLearningPolicyNetwork:
    def __init__(self, model):
        self.model = model

    def _create_x(self, player):
        board, is_black, putable_position_nums = player

        mine    = np.array([1 if (is_black and v == 1) or (not is_black and v == -1) else 0 for v in board], dtype=np.float32)
        yours   = np.array([1 if (is_black and v == -1) or (not is_black and v == 1) else 0 for v in board], dtype=np.float32)
        blank   = np.array([1 if v == 0 else 0 for v in board], dtype=np.float32)
        putable = np.array([1 if i in putable_position_nums else 0 for i in range(64)], dtype=np.float32)

        # 64 + 64 + 64 + 64
        x = np.concatenate((mine, yours, blank, putable)).reshape((1, 4, 8, 8))
        return x

    def get_policy_and_value(self, player):
        policy, value = self.model(self._create_x(player))
        return policy, value

    def __call__(self, player):
        _, _, putable_position_nums = player

        policy, _ = self.get_policy_and_value(player)

        putable_position_probabilities = np.array([policy[0].data[num] for num in putable_position_nums])
        indexs = np.where(putable_position_probabilities == putable_position_probabilities.max())[0]
        index = np.random.choice(indexs)
        choice = putable_position_nums[index]
        choice_data = { 'position_num': choice }
        return choice_data

class ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch:
    def __init__(self, model):
        self.model = model
        self.sl = ChoiceSupervisedLearningPolicyNetwork(self.model)

    def _get_node(self, player, position_num, probability):
        return {
            'player': player,
            'position_num': position_num,
            'try_num': 0,
            'win_num': 0,
            'probability': probability,
            'value': None,
            'child_nodes': None
        }

    def _get_initial_nodes(self, player):
        board, is_black, putable_position_nums = player

        policy, value = self.sl.get_policy_and_value(player)

        putable_position_probabilities = np.array([policy[0].data[num] for num in putable_position_nums])
        putable_position_probabilities /= putable_position_probabilities.sum()

        v = value[0][0].data

        nodes = [self._get_node(get_player(put(player, position_num), not is_black), position_num, putable_position_probabilities[i]) for i, position_num in enumerate(putable_position_nums)]
        if len(putable_position_nums) == 0:
            nodes = [self._get_node(get_player(board, not is_black), None, 1.0)]

        return v, nodes

    def _get_score(self, node, total_num):
        return (node['win_num'] / (1 + node['try_num'])) + node['probability'] * (math.sqrt(total_num) / (1 + node['try_num']))

    def _selection_node_index(self, nodes):
        total_num = functools.reduce(lambda total_num, node: total_num + node['try_num'], nodes, 0)
        scores = np.array([self._get_score(node, total_num) for node in nodes])
        indexs = np.where(scores == scores.max())[0]
        index = np.random.choice(indexs)
        return index

    def _selection_expansion(self, nodes):
        game = []
        node, path = None, []
        target_nodes = nodes
        while True:
            index = self._selection_node_index(target_nodes)
            path.append(index)
            node = target_nodes[index]
            if node['child_nodes'] is None:
                # expansion
                value, child_nodes = self._get_initial_nodes(node['player'])
                node['value'] = value
                if not is_end_game(game, node['player']):
                    node['child_nodes'] = child_nodes
                break
            target_nodes = node['child_nodes']
            choice_data = { 'position_num': node['position_num'] }
            game.append((node['player'], choice_data))
        return nodes, node, path

    def _backup(self, nodes, path, value):
        target_nodes = nodes
        for index in path:
            target_nodes[index]['try_num'] += 1
            target_nodes[index]['win_num'] += value
            target_nodes = target_nodes[index]['child_nodes']
        return nodes

    def _choice_node_index(self, nodes):
        try_nums = np.array([node['try_num'] for node in nodes])
        indexs = np.where(try_nums == try_nums.max())[0]
        index = np.random.choice(indexs)
        return index

    def __call__(self, player, try_num = 1500):
        _, nodes = self._get_initial_nodes(player)
        for _ in range(try_num):
            nodes, node, path = self._selection_expansion(nodes)
            nodes = self._backup(nodes, path, node['value'])
        index = self._choice_node_index(nodes)
        choice = nodes[index]['position_num']
        candidates = [{ 'position_num': "{}".format(node['position_num']), 'try_num': "{}".format(node['try_num']) } for node in nodes]
        choice_data = { 'position_num': choice, 'candidates': candidates }
        return choice_data

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
    choice_data = { 'position_num': choice }
    return choice_data

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
            choice_data = choice(player)
            position_num = choice_data["position_num"]
            if is_render:
                print(position_num)
            board = put(player, position_num)
        else:
            if is_render:
                print("pass")
        game.append((player, choice_data))
        player = get_player(board, not is_black)
    return game

def save_playdata(steps):
    playdata = []
    for step in steps:
        _, choice_data = step
        position_num = choice_data['position_num']
        playdata.append({"position_num": "{}".format(position_num if position_num is not None else -1)})
    filename = 'data/playdata_{}.dat'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(filename, 'a') as f:
        f.write("{}\n".format(json.dumps(playdata)))

def save_self_playdata(steps):
    self_playdata = []
    for step in steps:
        _, choice_data = step
        position_num = choice_data['position_num']
        self_playdata.append({
            "position_num": "{}".format(position_num if position_num is not None else -1),
            'candidates': choice_data['candidates']
        })
    filename = 'data/self_playdata_{}.dat'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(filename, 'a') as f:
        f.write("{}\n".format(json.dumps(self_playdata)))

def play():
    while True:
        steps = game(choice_human, ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(DualNet()))
        save_playdata(steps)

def self_play():
    model1 = DualNet()
    model2 = DualNet()
    steps = game(
        ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(model1),
        ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(model2)
    )
    save_self_playdata(steps)

def replay(steps_list):
    for steps in steps_list:
        position_nums = [int(step['position_num']) for step in json.loads(steps)]
        choice1 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[::2])
        choice2 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[1::2])
        game(choice1, choice2)

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1 and args[1] == 'play':
        play()
    elif len(args) > 2 and args[1] == 'replay':
        with open(args[2], 'r') as f:
            steps_list = f.readlines()
            replay(steps_list)
    elif len(args) > 1 and args[1] == 'selfplay':
        self_play()
    else:
        print('Usage error:', file=sys.stderr)
        print(' - python main.py play', file=sys.stderr)
        print(' - python main.py replay filepath-playdata', file=sys.stderr)
        print(' - python main.py selfplay', file=sys.stderr)
