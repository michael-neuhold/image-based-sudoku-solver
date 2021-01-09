from typing import List, Tuple
import cv2
import numpy as np
import os

kernel7 = (
    np.array([
        [ 0, 0, 1, 1, 1, 0, 0 ],
        [ 0, 1, 1, 1, 1, 1, 0 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 0, 1, 1, 1, 1, 1, 0 ],
        [ 0, 0, 1, 1, 1, 0, 0 ],
    ], dtype=np.uint8))

kernel5 = (
    np.array([
        [ 0, 1, 1, 1, 0 ],
        [ 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1 ],
        [ 0, 1, 1, 1, 0 ]
    ], dtype=np.uint8))

kernel3 = (
    np.array([
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
    ], dtype=np.uint8))


def downsample(img, fy, debug_output=None, debug_filename = None):
    height, width = img.shape
    fx = int( round((fy/height) * width) )

    # downsample
    downsampled = cv2.resize(img, # original image
                    (fx, fy), # set fx and fy, not the final size
                    interpolation=cv2.INTER_AREA)

    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'downsampled.jpg'), downsampled)

    return downsampled, (fy / height)


def normalize(img, debug_output=None, debug_filename = None):
    float_gray = img.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=5, sigmaY=5)
    num = float_gray - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=51, sigmaY=51)
    den = cv2.pow(blur, 0.5)

    gray = num / (den + 1)

    gray = cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX) * 255
    norm = gray.astype(np.uint8)

    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'norm.jpg'), norm)

    return norm


def get_edges(img, debug_output=False, debug_filename = None):
    canny = cv2.Canny(img, 200, 380)

    # join edges => every line consists of 2 edges
    result = cv2.dilate(canny, kernel7, iterations=1)

    # remove thin edges
    result = cv2.erode(result, kernel7, iterations=1)
    result = cv2.erode(result, kernel3, iterations=1)

    # regrow remaining edges
    result = cv2.dilate(result, kernel3, iterations=1)
    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'edges.jpg'), result)
    return result




def get_biggest_connected_component(edge_img, debug_output=None, debug_filename = None) -> Tuple:
    # calculate components
    component_count, component_img, stats, centroids = ( 
        cv2.connectedComponentsWithStats(edge_img, connectivity=8))

    # generate list of '(id, size) - tuples'
    id_area_pairs = list(
        #              id,   area (= 'size')
        map(lambda x: (x[0], x[1][cv2.CC_STAT_AREA]),
            enumerate(stats)))

    # switched from sorted to max
    # find largest component (~ background)
    bg_stat = max(id_area_pairs, key=lambda stats: stats[1])
    bg_id = bg_stat[0]

    # find second largest component (~ sudoku)
    component_stat = max(id_area_pairs, key=lambda stat: stat[1] if stat[0] != bg_id else 0)
    component_id = component_stat[0]
    component_size = component_stat[1]
    
    # select biggest component
    component = cv2.inRange(component_img, component_id, component_id)
    
    debug_output and cv2.imwrite(os.path.join(debug_output, debug_filename or 'component.jpg'), component)

    return component, component_size

