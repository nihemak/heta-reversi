from flask import Flask, render_template, request
from Reversi import Reversi
import base64
import json
import numpy as np

app = Flask(__name__)
app.jinja_env.globals.update(chr=chr)
app.jinja_env.globals.update(int=int)
app.jinja_env.globals.update(base64=base64)
app.jinja_env.globals.update(json=json)

@app.route('/')
def index():
    board = Reversi.get_init_board()
    query = request.args.get('query')
    if query:
        params = json.loads(base64.urlsafe_b64decode(query).decode())
        player = Reversi.get_player(np.array(params['board']))
        board = Reversi.put(player, params['num'])
        while True:
            player = Reversi.get_player(board, False)
            if Reversi.is_putable(player):
                _, _, putable_position_nums = player
                board = Reversi.put(player, np.random.choice(putable_position_nums))
                player = Reversi.get_player(board)
                if Reversi.is_putable(player):
                    break
            else:
                break

    board, is_black, putable_position_nums = Reversi.get_player(board)
    black_num, white_num = Reversi.get_stone_num(board)
    return render_template('index.html',
                           is_end_board=Reversi.is_end_board(board), black_num=black_num, white_num=white_num, board=board, is_black=is_black, putable_position_nums=putable_position_nums)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')