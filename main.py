# coding: utf-8

import numpy as np
import json
import datetime
import sys
import boto3

from Reversi import choice_random, Reversi
from Reversi import DualNet, ChoiceReplaySteps, choice_primitive_monte_carlo, ChoiceMonteCarloTreeSearch, ChoiceSupervisedLearningPolicyNetwork, ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch, DualNetTrainer, choice_human

def save_playdata(steps):
    playdata = []
    for step in steps:
        _, choice_data = step
        position_num = choice_data['position_num']
        playdata.append({"position_num": "{}".format(position_num if position_num is not None else -1)})
    filename = 'data/playdata_{}.dat'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(filename, 'a') as f:
        f.write("{}\n".format(json.dumps(playdata)))

def yes_no_input(message):
    yes = False
    while True:
        try:
            choice = input(message).lower()
            if choice in ['y', 'ye', 'yes']:
                yes = True
                break
            elif choice in ['n', 'no']:
                break
            else:
                print("{} is invalid".format(choice))
        except Exception:
            print("{} is invalid".format(choice))
    return yes

def play(choice_computer):
    choice1 = {
        'name': "You",
        'choice': choice_human
    }
    choice2 = {
        'name': "Computer",
        'choice': choice_computer
    }
    while True:
        print("start: {} vs {}".format(choice1['name'], choice2['name']))

        steps = Reversi.game(choice1['choice'], choice2['choice'])

        is_black_win = Reversi.is_win_game(steps, is_black = True)
        is_white_win = Reversi.is_win_game(steps, is_black = False)
        if is_black_win:
            print("{} is win".format(choice1['name']))
        elif is_white_win:
            print("{} is win".format(choice2['name']))
        else:
            print("draw")
        save_playdata(steps)
        if not yes_no_input("Do you want to continue? [y/N]: "):
            break
        choice1, choice2 = choice2, choice1

def replay(steps_list):
    for steps in steps_list:
        position_nums = [int(step['position_num']) for step in json.loads(steps)]
        choice1 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[::2])
        choice2 = ChoiceReplaySteps(np.array(position_nums, dtype=np.int32)[1::2])
        Reversi.game(choice1, choice2)

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1 and args[1] == 'play-random':
        play(choice_random)
    elif len(args) > 1 and args[1] == 'play-primitive-monte-carlo':
        play(choice_primitive_monte_carlo)
    elif len(args) > 1 and args[1] == 'play-mcts':
        play(ChoiceMonteCarloTreeSearch())
    elif len(args) > 1 and args[1] == 'play-sl-policy-network-random':
        play(ChoiceSupervisedLearningPolicyNetwork(DualNet()))
    elif len(args) > 2 and args[1] == 'play-sl-policy-network':
        model = DualNet()
        model.load(args[2])
        play(ChoiceSupervisedLearningPolicyNetwork(model))
    elif len(args) > 1 and args[1] == 'play-apv-mcts-random':
        play(ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(DualNet()))
    elif len(args) > 2 and args[1] == 'play-apv-mcts':
        model = DualNet()
        model.load(args[2])
        play(ChoiceAsynchronousPolicyAndValueMonteCarloTreeSearch(model))
    elif len(args) > 2 and args[1] == 'replay':
        with open(args[2], 'r') as f:
            steps_list = f.readlines()
            replay(steps_list)
    elif len(args) > 1 and args[1] == 'create-model':
        trainer = DualNetTrainer()
        _, model_filename = trainer()
    elif len(args) > 2 and args[1] == 'create-model-batch':
        bucket_name = args[2]

        trainer = DualNetTrainer(
            self_play_try_num = 25,
            create_new_model_epoch_num = 10,
            evaluation_try_num = 40,
            evaluation_win_num = 22,
            try_num = 1,
            apv_mcts_try_num = 150,
            gpu_device = 0
        )
        _, model_filename = trainer()
        s3 = boto3.resource('s3')
        s3.Bucket(bucket_name).upload_file(model_filename, model_filename)
        print(model_filename)
    elif len(args) > 2 and args[1] == 'train-model':
        model = DualNet()
        model.load(args[2])
        trainer = DualNetTrainer(model)
        _, model_filename = trainer()
        print(model_filename)
    else:
        print('Usage error:', file=sys.stderr)
        print(' - python main.py play-random', file=sys.stderr)
        print(' - python main.py play-primitive-monte-carlo', file=sys.stderr)
        print(' - python main.py play-mcts', file=sys.stderr)
        print(' - python main.py play-sl-policy-network-random', file=sys.stderr)
        print(' - python main.py play-sl-policy-network filepath-modeldata', file=sys.stderr)
        print(' - python main.py play-apv-mcts-random', file=sys.stderr)
        print(' - python main.py play-apv-mcts filepath-modeldata', file=sys.stderr)
        print(' - python main.py replay filepath-playdata', file=sys.stderr)
        print(' - python main.py create-model', file=sys.stderr)
        print(' - python main.py create-model-batch bucket-name', file=sys.stderr)
        print(' - python main.py train-model filepath-modeldata', file=sys.stderr)
