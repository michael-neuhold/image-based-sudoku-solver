from math import pi
from typing import Tuple
import cv2
import numpy as np


if __name__ != '__main__': 
    from sudoku_lib.utils import prep, lines
    from sudoku_lib.sudoku_cython import calc_component_bound

kernel3 = (
    np.array([
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
    ], dtype=np.uint8))

def extract_sudoku_component(img, debug_output=None, debug_filename=None) -> Tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray, scalef1 = prep.downsample(gray, 480)
    norm = prep.normalize(gray)
    edge_img  = prep.get_edges(norm)

    component_img, component_size = prep.get_biggest_connected_component(edge_img)
    # eroded = cv2.erode(component_img, kernel3, iterations=1)

    result, scalef2 = prep.downsample(component_img, 240)
    _, result = cv2.threshold(result, 64, 255, cv2.THRESH_BINARY)
    
    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'component.jpg'), result)
    scalef = scalef1 * scalef2

    return result, component_size * scalef * scalef, scalef


def unwarp(oriented_corners, scalef, img, debug_output=None, debug_filename=None):
    pts2 = np.float32([
                [576-1, 576-1],
                [0, 576-1],
                [0, 0], 
                [576-1, 0],
            ]) 
    pts1 = np.float32([
        *map(lambda p: [(p[0] + 0.5) / scalef, (p[1] + 0.5) / scalef], oriented_corners)
    ]) 
        
    # Apply Perspective Transform Algorithm 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1) 

    unwarped = cv2.warpPerspective(img, matrix, (576, 576))

    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'done.jpg'), unwarped)

    return unwarped, inv_matrix


def render_lines(img, lines, scalef, color=(255,255,255)):
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

        cv2.line(img, (x1, y1), (x2, y2), color, 1)
        # print(f'theta {theta} | rho {rho} | ({x0}, {y0})')
    
    # cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'applied_hough.jpg'), img)


def render_corners(img, corners, scalef):
    cv2.circle(img, (int((corners[0][0]+0.5)/scalef), int((corners[0][1]+0.5)/ scalef)), 3, color=(255,0,0), thickness=5)
    cv2.circle(img, (int((corners[1][0]+0.5)/scalef), int((corners[1][1]+0.5)/ scalef)), 3, color=(0,255,0), thickness=5)
    cv2.circle(img, (int((corners[2][0]+0.5)/scalef), int((corners[2][1]+0.5)/ scalef)), 3, color=(0,0,255), thickness=5)
    cv2.circle(img, (int((corners[3][0]+0.5)/scalef), int((corners[3][1]+0.5)/ scalef)), 3, color=(0,0,0), thickness=5)

    # cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'corner_points.jpg'), img)

def render_bound(img, corners, scalef):
    for i in range(len(corners)):
        j = (i + 1) % 4

        x1 = int((corners[i][0] + 0.5) / scalef)
        y1 = int((corners[i][1] + 0.5) / scalef)
        x2 = int((corners[j][0] + 0.5) / scalef)
        y2 = int((corners[j][1] + 0.5) / scalef)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# import sudoku_cython
def extract(input_img, debug_stage=None):
    component, component_size, scalef = extract_sudoku_component(input_img)

    # calc_component_bound
    bound_img_ = calc_component_bound(component, np.zeros(component.shape, dtype='uint8'))
    bound_img = np.array(bound_img_).astype(np.uint8)

    if debug_stage == 'component':
        print(f'component_size = {component_size}')
        return component


    # apply HoughLines
    bound_pixel_count =  np.sum(bound_img) / 255
    hough_threshold = int(bound_pixel_count / 4 * 0.45)
    # print(f'hough_threshold = {hough_threshold}')
    hough_lines = cv2.HoughLines(bound_img, rho=1, theta=np.pi/360, threshold=hough_threshold)
    if debug_stage == 'hough':
        if not (hough_lines is None):
            render_lines(input_img, hough_lines, scalef)
            # print(f'len(hough_lines) = {len(hough_lines)}')
        return input_img

    
    filtered_lines = lines.filter_outliers(hough_lines)
    filtered_lines = lines.filter_similar_new(filtered_lines, component.shape[1], component.shape[0])

    if debug_stage == 'hough-filtered':
        render_lines(input_img, filtered_lines, scalef, (0, 255, 0))
        return input_img

    if len(filtered_lines) == 0:
        return None

    horizontal_lines, vertical_lines = lines.split_horizontal_vertical(filtered_lines)
        
    oriented_corners = lines.calc_oriented_corners(horizontal_lines, vertical_lines)

    if len(oriented_corners) != 4:
        return None

    if debug_stage == 'corners':
        render_corners(input_img, oriented_corners, scalef)
        return input_img

    if debug_stage == 'bound':
        render_bound(input_img, oriented_corners, scalef)
        return input_img

    unwarped = unwarp(oriented_corners, scalef, input_img)
    return unwarped


def extract_with_bound(input_img):
    component, component_size, scalef = extract_sudoku_component(input_img)

    # calc_component_bound
    bound_img_ = calc_component_bound(component, np.zeros(component.shape, dtype='uint8'))
    bound_img = np.array(bound_img_).astype(np.uint8)
    bound_pixel_count =  np.sum(bound_img) / 255

    # apply HoughLines
    hough_threshold = int(bound_pixel_count / 4 * 0.45)
    hough_lines = cv2.HoughLines(bound_img, rho=1, theta=np.pi/360, threshold=hough_threshold)
    
    filtered_lines = lines.filter_outliers(hough_lines)
    filtered_lines = lines.filter_similar_new(filtered_lines, component.shape[1], component.shape[0])

    if len(filtered_lines) == 0:
        return None

    horizontal_lines, vertical_lines = lines.split_horizontal_vertical(filtered_lines)
    oriented_corners = lines.calc_oriented_corners(horizontal_lines, vertical_lines)

    if len(oriented_corners) != 4:
        return None

    unwarped, warp_matrix = unwarp(oriented_corners, scalef, input_img)
    render_bound(input_img, oriented_corners, scalef)

    return unwarped, warp_matrix



if __name__ == '__main__':
    import os
    import time
    import argparse

    from utils import prep, lines


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

    active_ex = active_ex or 0
    img = cv2.imread(
            os.path.join(input_dir, examples[active_ex]))
    original = img.copy()

    start = time.time()

    component, component_size, scalef = extract_sudoku_component(img, DEBUG_OUTPUT)

    # apply HoughLines
    hough_lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=100)
    
    filtered_lines = lines.filter_similar_new(hough_lines, component.shape[1], component.shape[0], DEBUG_OUTPUT)

    # split into vertical and horizontal lines
    horizontal_lines, vertical_lines = lines.split_horizontal_vertical(filtered_lines)

    DEBUG_OUTPUT and render_lines(img, filtered_lines, scalef)
        
    oriented_corners  = lines.calc_oriented_corners(horizontal_lines, vertical_lines)

    DEBUG_OUTPUT and render_corners(img, oriented_corners, scalef)

    unwarp(oriented_corners, scalef, original, DEBUG_OUTPUT)

    end = time.time()
    print(f'computation time: {(end - start)*1000}ms')




