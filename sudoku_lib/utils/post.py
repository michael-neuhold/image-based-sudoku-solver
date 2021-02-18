from typing import Tuple
import cv2
import numpy as np

DIGIT_WIDTH = 64
DETECTOR_MARGIN = 16
CNN_INPUT_MARGIN = 4

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

