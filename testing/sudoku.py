from math import pi
from typing import Tuple
import cv2
import numpy as np
import prep
import lines



def extract_sudoku_component(img, debug_output=None, debug_filename=None) -> Tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray, scalef1 = prep.downsample(gray, 480)
    norm = prep.normalize(gray)
    edge_img  = prep.get_edges(norm)

    component_img, component_size = prep.get_biggest_connected_component(edge_img)

    result, scalef2 = prep.downsample(component_img, 240)
    _, result = cv2.threshold(result, 110, 255, cv2.THRESH_BINARY)
    
    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'component.jpg'), result)
    scalef = scalef1 * scalef2

    return result, component_size, scalef


def unwarp(oriented_corners, scalef, img, debug_output=None, debug_filename=None):
    pts2 = np.float32([
                [512-1, 512-1],
                [0, 512-1],
                [0, 0], 
                [512-1, 0],
            ]) 
    pts1 = np.float32([
        *map(lambda p: [(p[0] + 0.5) / scalef, (p[1] + 0.5) / scalef], oriented_corners)
    ]) 
        
    # Apply Perspective Transform Algorithm 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    unwarped = cv2.warpPerspective(img, matrix, (512, 512))

    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'done.jpg'), unwarped)


def render_lines(img, lines):
    # render lines (debug)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int((x0 + 1000*(-b) + 0.5) / scalef)
        y1 = int((y0 + 1000*(a) + 0.5) / scalef)
        x2 = int((x0 - 1000*(-b) + 0.5) / scalef)
        y2 = int((y0 - 1000*(a) + 0.5) / scalef)

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # print(f'theta {theta} | rho {rho} | ({x0}, {y0})')
    
    cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'applied_hough.jpg'), img)


def render_corners(img, corners):
    cv2.circle(img, (int((corners[0][0]+0.5)/scalef), int((corners[0][1]+0.5)/ scalef)), 3, color=(255,0,0), thickness=5)
    cv2.circle(img, (int((corners[1][0]+0.5)/scalef), int((corners[1][1]+0.5)/ scalef)), 3, color=(0,255,0), thickness=5)
    cv2.circle(img, (int((corners[2][0]+0.5)/scalef), int((corners[2][1]+0.5)/ scalef)), 3, color=(0,0,255), thickness=5)
    cv2.circle(img, (int((corners[3][0]+0.5)/scalef), int((corners[3][1]+0.5)/ scalef)), 3, color=(0,0,0), thickness=5)

    cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'corner_points.jpg'), img)


def extract(input_img):
    component, component_size, scalef = extract_sudoku_component(input_img)

    # apply HoughLines
    hough_lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=100)
    filtered_lines = lines.filter_similar(hough_lines)

    if len(filtered_lines) == 0:
        return None

    horizontal_lines, vertical_lines = lines.split_horizontal_vertical(filtered_lines)
        
    oriented_corners = lines.calc_oriented_corners(horizontal_lines, vertical_lines)

    if len(oriented_corners) != 4:
        return None

    unwarped = unwarp(oriented_corners, scalef, input_img)
    return unwarped



if __name__ == '__main__':
    import os
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int)
    args = parser.parse_args()

    active_ex = args.example

    # read image
    dirname  = os.path.dirname(__file__)
    input_dir = os.path.join(dirname, '..', 'images', 'input')
    output_dir = os.path.join(dirname, '..', 'images', 'output')

    DEBUG_OUTPUT = output_dir

    examples= [
        'sudoku_001.jpg',
        'sudoku_010.jpg',
        'sudoku_002.jpg',
        'sudoku_003.jpg',
        'sudoku_004.jpg',
        'sudoku_005.jpg',
        'sudoku_006.jpg',
        'sudoku_007.jpg'
    ]

    img = cv2.imread(
            os.path.join(input_dir, examples[active_ex]))
    original = img.copy()

    start = time.time()

    component, component_size, scalef = extract_sudoku_component(img, DEBUG_OUTPUT)

    # apply HoughLines
    hough_lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=100)
    filtered_lines = lines.filter_similar(hough_lines, DEBUG_OUTPUT)

    # split into vertical and horizontal lines
    horizontal_lines, vertical_lines = lines.split_horizontal_vertical(filtered_lines)

    DEBUG_OUTPUT and render_lines(img, filtered_lines)
        
    oriented_corners  = lines.calc_oriented_corners(horizontal_lines, vertical_lines)

    DEBUG_OUTPUT and render_corners(img, oriented_corners)

    end = time.time()
    print(end - start)

    unwarp(oriented_corners, scalef, original, DEBUG_OUTPUT)




