class SudokuDetectorState:
    def __init__(self):
        self.sudoku_solved = False
        self.sudoku_result = None
        self.empty_pos = []
        self.empty_counter = 0