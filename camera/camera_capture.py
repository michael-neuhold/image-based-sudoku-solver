from PyQt5.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import sys

from sudoku_lib import sudoku

def displayFrame():
    ret, frame = cap.read()
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
    # print input image
    inputImageBox.setPixmap(QPixmap.fromImage(p))

    unwarped = sudoku.extract(frame)
    if not (unwarped is None):
        output = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
        h, w, ch = output.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(output.data, w, h, bytesPerLine, QImage.Format_RGB888)
        outputImageBox.setPixmap(QPixmap.fromImage(convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)))
    else:
        pass # outputImageBox.setPixmap(QPixmap.fromImage(p))


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
