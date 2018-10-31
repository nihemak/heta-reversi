# coding: utf-8

def render_board(board_data):
    row = " {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} "
    hr = "\n------------------------------------------------\n"
    layout = row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row
    print((layout).format(*board_data))

def main():
    board = range(64)
    render_board(board)

if __name__ == "__main__":
    main()
