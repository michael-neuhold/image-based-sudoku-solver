from sudoku_lib import sudoku
# from digit_recognition import digit


MODE = 'DEBUG' # 'DEBUG' | 'RELEASE'


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

def extract_numbers(frame) -> []:
    # extract sudoku
    unwarped = sudoku.extract(frame)

    # cv2.imwrite(os.path.join(output_dir, 'test.jpg'), unwarped)
    # return []

    digits = []
    if not (unwarped is None):
        # extract individual digits
        for y in range(9):
            for x in range(9):
                # determine if tile contains digit
                stddev = (
                    calc_stddev(
                        extract_detector_region(unwarped, (x, y))))

                # stddev:
                # - empty: [23, 114]
                # - digit: [1248, 1583]

                if stddev >= 200: # contains digit
                    cnn_input = (
                        post_process_cnn_input(
                            extract_cnn_input(unwarped, (x, y))))
                    digits.append(cnn_input)
    
    return digits


import os

sudoku_images = [
        'sudoku_001.jpg',
        'sudoku_002.jpg',
        'sudoku_003.jpg',
        'sudoku_004.jpg',
        'sudoku_005.jpg',