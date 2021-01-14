from PyQt5.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import sys

from sudoku_lib import sudoku
from digit_recognition import digit

COMPONENT = 'component'
HOUGH = 'hough'
HOUGH_FILTERED = 'hough-filtered'
CORNERS = 'corners'
BOUND = 'bound'

DIGIT_WIDTH = 64
DETECTOR_MARGIN = 16
CNN_INPUT_MARGIN = 8

x_check = 8
y_check = 2

def displayFrame():
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
        for y in range(9):
            digits.append([])
            for x in range(9):
                # determine if tile contains digit
                digit_frame = unwarped[ y*DIGIT_WIDTH + DETECTOR_MARGIN : (y+1)*DIGIT_WIDTH - DETECTOR_MARGIN, 
                                        x*DIGIT_WIDTH + DETECTOR_MARGIN : (x+1)*DIGIT_WIDTH - DETECTOR_MARGIN ]
                digit_frame = cv2.cvtColor(digit_frame, cv2.COLOR_BGR2GRAY)
                digit_frame = digit_frame.astype('float32')
                avg = np.sum(digit_frame) / ((DIGIT_WIDTH - 2*DETECTOR_MARGIN)**2)
                shifted = digit_frame - avg
                squared = shifted * shifted
                stddev = np.sqrt(np.sum(squared))

                # stddev:
                # - empty: [23, 114]
                # - digit: [1248, 1583]

                if stddev < 200: # empty
                    print(' ', end='')
                    digits[y].append(None)
                else:            # contains digit
                    print('#', end='')
                    cnn_input = unwarped[ y*DIGIT_WIDTH + CNN_INPUT_MARGIN : (y+1)*DIGIT_WIDTH - CNN_INPUT_MARGIN, 
                                          x*DIGIT_WIDTH + CNN_INPUT_MARGIN : (x+1)*DIGIT_WIDTH - CNN_INPUT_MARGIN ]
                    cnn_input = cv2.cvtColor(cnn_input, cv2.COLOR_BGR2GRAY)
                    cnn_input = cv2.resize(cnn_input, (28, 28), interpolation=cv2.INTER_AREA)
                    cnn_input = 255 - cnn_input
                    avg = np.sum(cnn_input) / (28 * 28)
                    cnn_input = cnn_input.astype('float32')
                    cnn_input = (cnn_input - avg)*6 + 0
                    cnn_input = np.clip(cnn_input, 0, 255)
                    cnn_input = cnn_input.astype('uint8')
                
                    digits[y].append(cnn_input)

                    if y != y_check or x != x_check:
                        continue
                    else:
                        prediction = digit.predict(cnn_input)
                        print(prediction)
            print()

        print('------')
                
        if not (digits[y_check][x_check] is None):
            output = cv2.cvtColor(digits[y_check][x_check], cv2.COLOR_BGR2RGB)
                    # output = cv2.cvtColor(digits[5][5], cv2.COLOR_GRAY2RGB)
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
timer.timeout.connect(displayFrame)
timer.start(60)

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
