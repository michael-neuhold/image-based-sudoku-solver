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
MARGIN = 8

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
                digit_img = unwarped[ y*DIGIT_WIDTH + MARGIN : (y+1)*DIGIT_WIDTH - MARGIN, 
                                      x*DIGIT_WIDTH + MARGIN : (x+1)*DIGIT_WIDTH - MARGIN ]
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                digit_img = 255 - digit_img
                _, mask = cv2.threshold(digit_img, 180, 255, cv2.THRESH_BINARY)
                # mask =  cv2.erode(mask, np.ones((3,3)))                
                mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_AREA)
                
                digits[y].append(mask)

                if y != 5 or x != 7:
                   continue
                else:
                   prediction = digit.predict(mask)
                   print(np.argmax(prediction))
                

        output = cv2.cvtColor(digits[5][7], cv2.COLOR_BGR2RGB)
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
