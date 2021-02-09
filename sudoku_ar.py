from typing import Tuple
from PyQt5.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import sys
import os

from sudoku_lib import sudoku
from digit_recognition import digit


dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, 'images', 'input')
output_dir = os.path.join(dirname, 'images', 'output')

# number_id = 0
# def save_cnn_input(img):
#     global number_id
#     img_name = f'ml_input_{number_id:03}.png'
#     cv2.imwrite(os.path.join(output_dir, img_name), img)
#     number_id += 1



MODE = 'RELEASE' # 'DEBUG' | 'RELEASE'


COMPONENT = 'component'
HOUGH = 'hough'
HOUGH_FILTERED = 'hough-filtered'
CORNERS = 'corners'
BOUND = 'bound'

DIGIT_WIDTH = 64
DETECTOR_MARGIN = 16
CNN_INPUT_MARGIN = 4

x_check = 8
y_check = 2

def post_process_cnn_input(raw):
    inverted = 255 - raw
    avg = np.sum(inverted) / (28 * 28)
    intermediate = inverted.astype('float32')
    intermediate = (intermediate - avg)*6 + 0
    processed = np.clip(intermediate, 0, 255)
    processed = processed.astype('uint8')
    return processed

def extract_cnn_input(input_img, tile: Tuple):
    x, y = tile
    cnn_input = input_img[ y*DIGIT_WIDTH + CNN_INPUT_MARGIN : (y+1)*DIGIT_WIDTH - CNN_INPUT_MARGIN, 
                           x*DIGIT_WIDTH + CNN_INPUT_MARGIN : (x+1)*DIGIT_WIDTH - CNN_INPUT_MARGIN ]
    cnn_input = cv2.cvtColor(cnn_input, cv2.COLOR_BGR2GRAY)
    cnn_input = cv2.resize(cnn_input, (28, 28), interpolation=cv2.INTER_AREA)
    return  cnn_input

def extract_detector_region(input_img, tile: Tuple):
    x, y = tile
    detector_region = input_img[ y*DIGIT_WIDTH + DETECTOR_MARGIN : (y+1)*DIGIT_WIDTH - DETECTOR_MARGIN, 
                                 x*DIGIT_WIDTH + DETECTOR_MARGIN : (x+1)*DIGIT_WIDTH - DETECTOR_MARGIN ]
    detector_region = cv2.cvtColor(detector_region, cv2.COLOR_BGR2GRAY)
    return detector_region

def calc_stddev(input_img): # grey 8bit
    img = input_img.astype('float32')
    avg = np.sum(img) / ((DIGIT_WIDTH - 2*DETECTOR_MARGIN)**2)
    shifted = img - avg
    squared = shifted * shifted
    stddev = np.sqrt(np.sum(squared))
    return stddev

def display_frame():
    ret, frame = cap.read()

    # extract sudoku
    unwarped = sudoku.extract_with_bound(frame)

    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
    # print input image
    inputImageBox.setPixmap(QPixmap.fromImage(p))

    if not (unwarped is None):
        # extract individual digits
        digits = []
        digit_pos = []
        for y in range(9):
            for x in range(9):
                # determine if tile contains digit
                stddev = (
                    calc_stddev(
                        extract_detector_region(unwarped, (x, y))))

                # stddev:
                # - empty: [35, 145]
                # - digit: [1458, 2103]

                if stddev >= 800: # contains digit
                    cnn_input = (
                        post_process_cnn_input(
                            extract_cnn_input(unwarped, (x, y))))
                
                    digits.append( cnn_input )
                    digit_pos.append( (x, y) )
                    # save_cnn_input(cnn_input)

        predictions = digit.predict_multiple(digits)
        reconstructed_sudoku = np.zeros((9, 9), dtype=np.int)
        for (value, (x, y)) in zip(predictions, digit_pos):
            reconstructed_sudoku[y, x] = value

        print(reconstructed_sudoku)
        print()

        # display unwarped
        output = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
        h, w, ch = output.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(output.data, w, h, bytesPerLine, QImage.Format_RGB888)
        outputImageBox.setPixmap(QPixmap.fromImage(convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)))

def display_frame_debug():
    ret, frame = cap.read()

    # extract sudoku
    copied = frame.copy()
    # _ = sudoku.extract(frame, HOUGH)
    unwarped = sudoku.extract(copied, BOUND)

    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
    # print input image
    inputImageBox.setPixmap(QPixmap.fromImage(p))

    if not (unwarped is None):
        output = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
        h, w, ch = output.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(output.data, w, h, bytesPerLine, QImage.Format_RGB888)
        outputImageBox.setPixmap(QPixmap.fromImage(convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)))


app = QApplication([])
window = QWidget()

# setup video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# setup timer
timer = QTimer()
if MODE == 'DEBUG':
    timer.timeout.connect(display_frame_debug)
elif MODE == 'RELEASE':
    timer.timeout.connect(display_frame)
timer.start(33) # 30fps

# setup ui elements
inputImageBox = QLabel('Input Image')
outputImageBox = QLabel('Output Image')
button = QPushButton("Exit")
button.clicked.connect(sys.exit) # quiter button 

# setup grid layout
grid = QGridLayout()
grid.addWidget(inputImageBox,0,0)
grid.addWidget(outputImageBox,0,1)
grid.addWidget(button, 1,0)

window.setLayout(grid)
window.show()
app.exec_()
