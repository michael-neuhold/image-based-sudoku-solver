def get_next_empty_position(board):
    """
    Returns the row and column of the next empty cell

    Parameter:
      board:[][]int The sudoku board to solve
      
    Returns:
      (int, int) The row and column of the next empty position
              or None, None if none was found
    """
    for row in range(9):
        for col in range(9):
            if(board[row][col] == 0):
                return row, col
    return None, None

def is_valid_row(board, row, guess):
    """
    Checks if guess is already in chosen row

    Parameter:
      board:[][]int The sudoku board to check
      row: int Row to check
      guess: The value to check for
      
    Returns:
      bool 
    """
    if(guess in board[row]):
        return False
    return True

def is_valid_col(board, col, guess):
    """
    Checks if guess is already in chosen column

    Parameter:
      board:[][]int The sudoku board to check
      col: int Column to check
      guess: The value to check for
      
    Returns:
      bool 
    """
    if guess in [board[i][col] for i in range(9)]:
        return False
    return True

def is_valid_box(board, row, col, guess):
    """
    Checks if guess is already in chosen box

    Parameter:
      board:[][]int The sudoku board to check
      row: int Row to check
      col: int Column to check
      guess: The value to check for
      
    Returns:
      bool 
    """

    first_col = (col // 3) * 3
    first_row = (row // 3) * 3

    for row in range(first_row, first_row + 3):
        for col in range(first_col, first_col + 3):
            if board[row][col] == guess:
                return False
    return True

def check_guess(board, row, col, guess):
    """
    Checks if guess is valid in board

    Parameter:
      board:[][]int The sudoku board to check
      row: int Row to check
      col: int Column to check
      guess: The value to check for
      
    Returns:
      bool 
    """
    return (is_valid_box(board, row, col, guess) and
            is_valid_col(board, col, guess) and
            is_valid_row(board, row, guess))

rec_count = 0
def solve_sudoku(board):
    """
    Solves given Sudoku via backtracking

    Parameter:
      board:[][]int The sudoku board to solve
      
    Returns:
      (bool,[][]int) The boolean indicates if a solution was found and 
                    the 2D field is the filled in sudoku 
    """
    global rec_count
    rec_count = 0
    return solve_sudoku_rec(board)

MAX_REC_DEPTH = 50000
def solve_sudoku_rec(board):
    """
    Recursively andvances trough sudoku to solve it.
    Maximum recursion depth can be adjusted trough MAX_REC_DEPTH

    Parameter:
      board:[][]int The sudoku board to solve
      
    Returns:
      (bool,[][]int) The boolean indicates if a solution was found and 
                    the 2D field is the filled in sudoku 
    """
    global rec_count
    rec_count += 1
    if (rec_count > MAX_REC_DEPTH):
        # end prematurely
        return (False, rec_count)

    row, col = get_next_empty_position(board)
    if row is None:
        return (True, rec_count)

    for guess in range(1, 10):

        if check_guess(board, row, col, guess):
            board[row][col] = guess
            if solve_sudoku_rec(board)[0]:
                return (True, rec_count)

    board[row][col] = 0

    return (False, rec_count)

def print_board(board):
    """
    Prints Sudoku

    Parameter:
      board:[][]int The sudoku board to print
    """
    print("\n-------------------------")
    for row in range(9):
        print("|", end="")
        for col in range(9):
            print(" " + str(board[row][col]) + "", end="")
            if((col + 1) % 3 == 0):
                print(" |", end="")
        if (row + 1) % 3 == 0:
            print("\n-------------------------")
        else:
            print()
			
if __name__ == '__main__':
    """
    Tests backtracking with hardcoded Sudoku
    """
    board = [
        [6, 0, 0, 0, 7, 1, 0, 0, 0],
        [0, 0, 0, 3, 0, 9, 2, 7, 0],
        [3, 0, 0, 2, 5, 6, 9, 0, 0],

        [0, 0, 8, 0, 3, 0, 7, 0, 0],
        [9, 0, 1, 0, 6, 4, 0, 0, 0],
        [0, 7, 0, 0, 9, 8, 0, 1, 0],

        [0, 1, 0, 6, 0, 0, 0, 0, 4],
        [0, 0, 0, 9, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 3]
    ]

    print_board(board)

    solved = solve_sudoku(board)

    print("\nsolved: ", solved)

    print_board(board)
