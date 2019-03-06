from flask import Flask, render_template
from Reversi import Reversi

app = Flask(__name__)
app.jinja_env.globals.update(chr=chr)
app.jinja_env.globals.update(int=int)

@app.route('/')
def index():
    board, is_black, putable_position_nums = Reversi.get_player(Reversi.get_init_board())
    return render_template('index.html', board=board, is_black=is_black, putable_position_nums=putable_position_nums)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')