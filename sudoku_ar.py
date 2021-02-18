from typing import Tuple
from PyQt5.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import sys

from sudoku_lib import sudoku
from sudoku_lib.utils import post as post_process
from sudoku_lib.utils.sudoku_detector_state import SudokuDetectorState
from digit_recognition import digit
from backtracking import sudoku_solver


def render_image(img):
    """
    displays an image on the renderTarget (= "main window")

    Parameter:
        img: [][]numpy_array the image to display
    """
    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
    p = convertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
    renderTarget.setPixmap(QPixmap.fromImage(p))


def find_digits(sudoku_img) -> Tuple:
    """
    iterates over a rectified/unwarped image of a sudoku
    and finds the digits and their positions in the sudoku.

    Parameter:
      sudoku_img: [][]numpy_array the image in which to search for digits

    Returns:
      ([ [][]numpy_array ], [ (int, int) ], [ (int, int) ]) 
        First: extracted sudoku digits
	    Second: positions of digits
	    Third: positions of empty sudoku-fields
    """
    
    digits = []
    digit_pos = []
    empty_pos = []

    for y in range(9):
        for x in range(9):
            # determine if tile contains digit
            stddev = (
                post_process.calc_stddev(
                    post_process.extract_detector_region(sudoku_img, (x, y))))

            # stddev:
            # - empty: [35, 145]
            # - digit: [1458, 2103]

            if stddev >= 800: # contains digit
                cnn_input = (
                    post_process.post_process_cnn_input(
                        post_process.extract_cnn_input(sudoku_img, (x, y))))
            
                digits.append( cnn_input )
                digit_pos.append( (x, y) )
            else:   # empty
                empty_pos.append( (x, y) )
    
    return digits, digit_pos, empty_pos


STATE = SudokuDetectorState()
def display_frame():
    global STATE

    ret, frame = cap.read()

    # extract sudoku
    extraction_result = sudoku.extract_with_bound(frame)
    display_image = frame

    if extraction_result is not None:
        STATE.empty_counter = 0

        unwarped, warp_matrix = extraction_result

        if not STATE.sudoku_solved:
            # extract individual digits
            digits, digit_pos, empty_pos = find_digits(unwarped)

            if len(digits) > 2:
                predictions = digit.predict_multiple(digits)
                reconstructed_sudoku = np.zeros((9, 9), dtype=np.int)
                for (value, (x, y)) in zip(predictions, digit_pos):
                    reconstructed_sudoku[y, x] = value

                print(reconstructed_sudoku)

                solve_succes, rec_count = sudoku_solver.solve_sudoku(reconstructed_sudoku)
                STATE.sudoku_solved = solve_succes
                STATE.sudoku_result = reconstructed_sudoku
                STATE.empty_pos = empty_pos

        else:
            blank_image = np.zeros(unwarped.shape, np.uint8)
            for (x, y) in STATE.empty_pos:
                x_ = int(x * (576 / 9) +12)
                y_ = int(y * (576 / 9) +40 +(8-y)*1)
                cv2.putText(blank_image, str(STATE.sudoku_result[y, x]), (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 6)

            warped = cv2.warpPerspective(blank_image, warp_matrix, frame.shape[0:2][::-1])

            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            bg = cv2.bitwise_and(frame, frame, mask = mask_inv)

            # Take only region of logo from logo image.
            fg = cv2.bitwise_and(warped,warped,mask = mask)

            # Put logo in ROI and modify the main image
            merged = cv2.add(bg, fg)

            display_image = merged

    else:  # no sudoku found
        STATE.empty_counter += 1
        if STATE.empty_counter > 10:
            STATE.sudoku_solved = False
            STATE.sudoku_result = None
            STATE.empty_pos = []

    render_image(display_image)



app = QApplication([])
window = QWidget()

# setup video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# setup timer
timer = QTimer()
timer.timeout.connect(display_frame)
timer.start(33) # 30fps

# setup ui elements
renderTarget = QLabel('Render-Target')
button = QPushButton("Exit")
button.clicked.connect(sys.exit) # quiter button 

# setup grid layout
grid = QGridLayout()
grid.addWidget(renderTarget,0,0)
grid.addWidget(button, 1,0)

window.setLayout(grid)
window.show()
app.exec_()
