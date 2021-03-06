# coding: utf-8

import numpy as np
import math
import functools
import json
import datetime
import uuid
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, serializers
from chainer.backends import cuda

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
        valid  = False
        while row >= 0 and row < 8 and column >= 0 and column < 8:
            position_num = row * 8 + column

            if exists == True and board[position_num] == own:
                valid = True
                break
            if board[position_num] != pair:
                break

            position_nums.append(position_num)

            exists = True
            row    += row_add
            column += column_add

        if valid == False:
            position_nums = []

        return position_nums

    @classmethod
    def get_flip_positions(cls, board, is_black, position_num):
        position_nums = []

        if not (position_num >= 0 and position_num <= 63):
            return position_nums
        if board[position_num] != 0:
            return position_nums

        column = position_num % 8
        row    = int(position_num / 8)

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

    def load(self, filename):
        serializers.load_npz(filename, self)

    def save(self, filename):
        serializers.save_npz(filename, self)

    @classmethod
    def get_input_data(cls, player):
        board, is_black, putable_position_nums = player

        mine    = np.array([1 if (is_black and v == 1) or (not is_black and v == -1) else 0 for v in board], dtype=np.float32)
        yours   = np.array([1 if (is_black and v == -1) or (not is_black and v == 1) else 0 for v in board], dtype=np.float32)
        blank   = np.array([1 if v == 0 else 0 for v in board], dtype=np.float32)
        putable = np.array([1 if i in putable_position_nums else 0 for i in range(64)], dtype=np.float32)

        # 64 + 64 + 64 + 64
        x = np.concatenate((mine, yours, blank, putable)).reshape((1, 4, 8, 8))
        return x

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

def choice_primitive_monte_carlo(player, try_num = 150):
    _, _, putable_position_nums = player
    position_scores = np.zeros(len(putable_position_nums))
    for _ in range(try_num):
        playouts = [Reversi.playout(player, position_num) for position_num in putable_position_nums]
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
        nodes = [self._get_node(Reversi.get_player(Reversi.put(player, position_num), not is_black), position_num) for position_num in putable_position_nums]
        if len(putable_position_nums) == 0:
            nodes = [self._get_node(Reversi.get_player(board, not is_black), None)]
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
                    if Reversi.is_end_game(game, node['player']):
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
        is_win = Reversi.playout(node['player'], node['position_num'])
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

    def __call__(self, player):
        _, _, putable_position_nums = player

        policy, _ = self.model(DualNet.get_input_data(player))

        putable_position_probabilities = np.array([policy[0].data[num] for num in putable_position_nums])
        indexs = np.where(putable_position_probabilities == putable_position_probabilities.max())[0]
        index = np.random.choice(indexs)
        choice = putable_position_nums[index]
        choice_data = { 'position_num': choice }
        return choice_data

class ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch:
    def __init__(self, model, is_strict_choice = True, try_num = 1500):
        self.model = model
        self.is_strict_choice = is_strict_choice
        self.default_params = {
            'try_num': try_num
        }

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

        policy, value = self.model(DualNet.get_input_data(player))

        putable_position_probabilities = np.array([policy[0].data[num] for num in putable_position_nums])
        putable_position_probabilities /= putable_position_probabilities.sum()

        v = value[0][0].data

        nodes = [self._get_node(Reversi.get_player(Reversi.put(player, position_num), not is_black), position_num, putable_position_probabilities[i]) for i, position_num in enumerate(putable_position_nums)]
        if len(putable_position_nums) == 0:
            nodes = [self._get_node(Reversi.get_player(board, not is_black), None, 1.0)]

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
                if not Reversi.is_end_game(game, node['player']):
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
        try_nums_sum = try_nums.sum()
        index = 0
        if self.is_strict_choice or try_nums_sum == 0:
            indexs = np.where(try_nums == try_nums.max())[0]
            index = np.random.choice(indexs)
        else:
            try_nums = try_nums.astype(np.float32)
            try_nums /= try_nums_sum  # to probability
            index = np.random.choice(range(len(try_nums)), p = try_nums)
        return index

    def __call__(self, player, try_num = None):
        try_num = self.default_params['try_num'] if try_num is None else try_num

        _, nodes = self._get_initial_nodes(player)
        for _ in range(try_num):
            nodes, node, path = self._selection_expansion(nodes)
            nodes = self._backup(nodes, path, node['value'])
        index = self._choice_node_index(nodes)
        choice = nodes[index]['position_num']
        candidates = [{ 'position_num': "{}".format(node['position_num']), 'try_num': "{}".format(node['try_num']) } for node in nodes]
        choice_data = { 'position_num': choice, 'candidates': candidates }
        return choice_data

class DualNetTrainer:
    def __init__(self, model = None, self_play_try_num = 2500, create_new_model_epoch_num = 100, evaluation_try_num = 400, evaluation_win_num = 220, try_num = 100, apv_mcts_try_num = 1500, gpu_device = -1):
        if model is None:
            model = DualNet()
        self.default_params = {
            'self_play_try_num': self_play_try_num,
            'create_new_model_epoch_num': create_new_model_epoch_num,
            'evaluation_try_num': evaluation_try_num,
            'evaluation_win_num': evaluation_win_num,
            'try_num': try_num,
            'apv_mcts_try_num': apv_mcts_try_num
        }
        self.gpu_device = gpu_device
        self._set_model(model)

    def _set_model(self, model):
        self.model = model
        date_str   = datetime.date.today().strftime("%Y%m%d")
        unique_str = str(uuid.uuid4())
        self.model_filename = 'data/model_{}_{}.dat'.format(date_str, unique_str)
        self.model.save(self.model_filename)
        print("[set model] model_filename: {}".format(self.model_filename))

    def _save_self_playdata(self, steps, filename):
        self_playdata = []
        is_black_win = Reversi.is_win_game(steps, is_black = True)
        is_white_win = Reversi.is_win_game(steps, is_black = False)

        is_black = True
        for step in steps:
            _, choice_data = step
            position_num = choice_data['position_num']
            win_score = -1
            if is_black_win and is_white_win:
                win_score = 0
            elif (is_black and is_black_win) or (not is_black and is_white_win):
                win_score = 1
            self_playdata.append({
                "position_num": "{}".format(position_num if position_num is not None else -1),
                "win_score": "{}".format(win_score),
                'candidates': choice_data['candidates']
            })
            is_black = not is_black
        with open(filename, 'a') as f:
            f.write("{}\n".format(json.dumps(self_playdata)))

    def _self_play(self, model1, model2, try_num = None, is_save_data = True, is_strict_choice = True):
        try_num = self.default_params['self_play_try_num'] if try_num is None else try_num

        player1 = {
            'is_model1': True,
            'choice': ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(model1, is_strict_choice, try_num = self.default_params['apv_mcts_try_num']),
            'win_num': 0
        }
        player2 = {
            'is_model1': False,
            'choice': ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(model2, is_strict_choice, try_num = self.default_params['apv_mcts_try_num']),
            'win_num': 0
        }
        date_str   = datetime.date.today().strftime("%Y%m%d")
        unique_str = str(uuid.uuid4())
        data_filename = 'data/self_playdata_{}_{}.dat'.format(date_str, unique_str)
        print("[self play] data_filename: {}".format(data_filename))
        for i in range(try_num):
            steps = Reversi.game(player1['choice'], player2['choice'], is_render = False)
            if Reversi.is_win_game(steps, is_black = True):
                player1['win_num'] += 1
            if Reversi.is_win_game(steps, is_black = False):
                player2['win_num'] += 1
            if is_save_data:
                self._save_self_playdata(steps, data_filename)
            player1, player2 = player2, player1
            (model1_player, model2_player) = (player1, player2) if player1['is_model1'] else (player2, player1)
            print("[self play] epoch: {} / {}, model1_win_num: {}, model2_win_num: {}".format(i + 1, try_num, model1_player['win_num'], model2_player['win_num']))
        (model1_win_num, model2_win_num) = (player1['win_num'], player2['win_num']) if player1['is_model1'] else (player2['win_num'], player1['win_num'])
        return model1_win_num, model2_win_num, data_filename

    def _get_train_y_policy(self, candidates, temperature = 0.5):
        y_policy = np.array([0] * 64, dtype=np.float32)
        sum_try_num = np.array([int(candidate['try_num']) ** (1 / temperature) for candidate in candidates]).sum()
        for candidate in candidates:
            y_policy[int(candidate['position_num'])] = (int(candidate['try_num']) ** (1 / temperature)) / sum_try_num
        return y_policy

    def _get_train_x(self, steps, step_num):
        position_nums = [int(step['position_num']) for step in steps]
        choice1 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[::2])
        choice2 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[1::2])
        steps = Reversi.game(choice1, choice2, is_render = False, limit_step_num = step_num)
        player, _ = steps[-1]
        x = DualNet.get_input_data(player)
        return x

    def _get_train_random(self, steps_list):
        steps      = json.loads(np.random.choice(steps_list))
        step_index = np.random.randint(0, len(steps) - 1)

        y_policy = self._get_train_y_policy(steps[step_index]['candidates'])
        y_value  = np.array([[int(steps[step_index]['win_score'])]], dtype=np.float32)
        x        = self._get_train_x(steps, (step_index + 1))
        return x, y_policy, y_value

    def _get_train_batch(self, steps_list, batch_size):
        batch_x, batch_y_policy, batch_y_value = [], [], []
        for _ in range(batch_size):
            x, y_policy, y_value = self._get_train_random(steps_list)
            batch_x.append(x)
            batch_y_policy.append(y_policy)
            batch_y_value.append(y_value)

        xp = np
        if self.gpu_device >= 0:
            xp = cuda.cupy

        x_train        = Variable(xp.array(batch_x)).reshape(-1, 4, 8, 8)
        y_train_policy = Variable(xp.array(batch_y_policy)).reshape(-1, 64)
        y_train_value  = Variable(xp.array(batch_y_value)).reshape(-1, 1)
        return x_train, y_train_policy, y_train_value

    def _create_new_model(self, steps_list, epoch_num = None, batch_size = 2048):
        epoch_num = self.default_params['create_new_model_epoch_num'] if epoch_num is None else epoch_num

        model = DualNet()
        model.load(self.model_filename)

        if self.gpu_device >= 0:
            cuda.get_device(self.gpu_device).use()
            model.to_gpu(self.gpu_device)

        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        for i in range(epoch_num):
            x_train, y_train_policy, y_train_value = self._get_train_batch(steps_list, batch_size)
            y_policy, y_value = model(x_train)
            model.cleargrads()
            loss = F.mean_squared_error(y_policy, y_train_policy) + F.mean_squared_error(y_value, y_train_value)
            loss.backward()
            optimizer.update()
            print("[new nodel] epoch: {} / {}, loss: {}".format(i + 1, epoch_num, loss))

        if self.gpu_device >= 0:
            model.to_cpu()

        return model

    def _evaluation(self, new_model, try_num = None, win_num = None):
        try_num = self.default_params['evaluation_try_num'] if try_num is None else try_num
        win_num = self.default_params['evaluation_win_num'] if win_num is None else win_num

        _, new_model_win_num, _ = self._self_play(self.model, new_model, try_num = try_num, is_save_data = False)
        if new_model_win_num >= win_num:
            self._set_model(new_model)

    def __call__(self, try_num = None):
        try_num = self.default_params['try_num'] if try_num is None else try_num

        for i in range(try_num):
            _, _, data_filename = self._self_play(self.model, self.model, is_strict_choice = False)
            steps_list = []
            with open(data_filename, 'r') as f:
                steps_list = f.readlines()
            new_model = self._create_new_model(steps_list)
            self._evaluation(new_model)
            print("[train] epoch: {} / {}".format(i + 1, try_num))
        return self.model, self.model_filename

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
