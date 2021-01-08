from math import atan2, pi, sqrt
from typing import Tuple
import cv2
import numpy as np
import os
import time

from numpy.core.fromnumeric import sort

active_ex = 3

import prep

# read image
dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, '..', 'images', 'input')
output_dir = os.path.join(dirname, '..', 'images', 'output')

DEBUG_OUTPUT = output_dir

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

def filter_lines(lines):
    if len(lines) == 0:
        print('No lines were found')
        return

    rho_threshold = 20
    theta_threshold = 0.17 # ~10°

    # how many lines are similar to a given one
    similar_lines = {i: [] for i in range(len(lines))}
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]

            diff_rad = theta_i - theta_j
            diff = abs(np.cos(diff_rad))

            if abs(abs(rho_i) - abs(rho_j)) < rho_threshold and diff > np.cos(16/180 * np.pi):
                similar_lines[i].append(j)
                similar_lines[j].append(i)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
        if not line_flags[indices[i]]:
            continue

        # we are only considering those elements that had less similar line
        for j in range(i + 1, len(lines)):
            # and only if we have not disregarded them already
            if not line_flags[indices[j]]:
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]

            diff_rad = theta_i - theta_j
            diff = abs(np.cos(diff_rad))

            if abs(abs(rho_i) - abs(rho_j)) < rho_threshold and diff > np.cos(16/180 * np.pi):
                # if it is similar and have not been disregarded yet then drop it now
                line_flags[indices[j]] = False

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    for i in range(len(lines)):  # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))

    return filtered_lines


examples= [
    'sudoku_001.jpg',
    'sudoku_004.jpg',
    'sudoku_006.jpg',
    'sudoku_007.jpg'
]


img = cv2.imread(
        os.path.join(input_dir, examples[active_ex]))

original = img.copy()


start = time.time()
component, component_size, scalef = extract_sudoku_component(img, DEBUG_OUTPUT)


print(component_size)


# apply HoughLines
lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=100)
filtered_lines = filter_lines(lines)





# filtered lines => corners

horizontal_lines = []
vertical_lines = []
for line1 in filtered_lines:
    rho0, theta0 = filtered_lines[0][0]
    rho1, theta1 = line1[0]

    # calc difference
    # 1 -> same
    # 0 -> 180° difference
    diff = theta1 - theta0
    diff = abs(np.cos(diff))

    if diff < np.cos(45/180 * np.pi):
        horizontal_lines.append(line1)
    else:
        vertical_lines.append(line1)

print(len(vertical_lines))
print(len(horizontal_lines))


# render lines (debug)
for line in filtered_lines:
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


DEBUG_OUTPUT and cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'applied_hough.jpg'), img)


# print image with highlighted, detected lines
edge_points = { 0: [], 1:[], 2: [], 3: [], 4: [] } #...

for hindex, hline in enumerate(horizontal_lines):
    points = []
    for vindex, vline in enumerate(vertical_lines):
        # switching v and h works for 002
        vrho, vtheta = vline[0]
        hrho, htheta = hline[0]

        # rh = x * cos(th) + y * sin(th)
        # rv = x * cos(tv) + y * sin(tv)
       
        # (rh - y * sin(th)) / cos(th) = x
        # rv = x * cos(tv) + y * sin(tv)


        # needs fix: something seems to be wrong 



        y = (vrho - (hrho * np.cos(vtheta) / np.cos(htheta))) / (np.sin(vtheta) - np.sin(htheta) * np.cos(vtheta) / np.cos(htheta))
        x = (hrho - y * np.sin(htheta)) / np.cos(htheta)

        
        # h, w = downsampled.shape
        # cv2.circle(img, (int(x/scalef), int(y/scalef)), 1, color=255, thickness=3)
        points.append((x, y, hindex, vindex))
        
    xsum = 0
    ysum = 0
    for point in points:
        xsum += point[0]
        ysum += point[1]
    xavg = xsum / len(points)
    yavg = ysum / len(points)

    xdsum = 0
    ydsum = 0
    for point in points:
        xd = point[0] - xavg
        xd = xd*xd
        yd = point[1] - yavg
        yd = yd*yd
        xdsum += xd
        ydsum += yd
    
    stdevy = sqrt(ydsum)
    stdevx = sqrt(xdsum)
    # print('stddev: ', stdevx, stdevy)

    min = points[0]
    max = points[0]
    if stdevx > stdevy:
        for point in points:
            if point[0] < min[0]:
                min = point
            if point[0] > max[0]:
                max = point
    else:
        for point in points:
            if point[1] < min[1]:
                min = point
            if point[1] > max[1]:
                max = point

    print('min: ', min)
    print('max: ', max)
    edge_points[min[3]].append(min)
    edge_points[max[3]].append(max)
    # cv2.circle(downsampled, (int(min[0]), int(min[1])), 1, color=255, thickness=3)
    # cv2.circle(downsampled, (int(max[0]), int(max[1])), 1, color=255, thickness=3)

DEBUG_OUTPUT and cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'applied_hough.jpg'), img)


# print(edge_points)

corner_points = []
for points in edge_points.values():
    if len(points) <= 0: continue

    xsum = 0
    ysum = 0
    for point in points:
        xsum += point[0]
        ysum += point[1]
    xavg = xsum / len(points)
    yavg = ysum / len(points)

    xdsum = 0
    ydsum = 0
    for point in points:
        xd = point[0] - xavg
        xd = xd*xd
        yd = point[1] - yavg
        yd = yd*yd
        xdsum += xd
        ydsum += yd
    
    stdevy = sqrt(ydsum)
    stdevx = sqrt(xdsum)
    print('stddev: ', stdevx, stdevy)

    min = points[0]
    max = points[0]
    if stdevx > stdevy:
        for point in points:
            if point[0] < min[0]:
                min = point
            if point[0] > max[0]:
                max = point
    else:
        for point in points:
            if point[1] < min[1]:
                min = point
            if point[1] > max[1]:
                max = point

    # cv2.circle(img, (int(min[0]/scalef), int(min[1]/ scalef)), 3, color=255, thickness=3)
    # cv2.circle(img, (int(max[0]/scalef), int(max[1]/scalef)), 3, color=255, thickness=3)
    corner_points.append(min)
    corner_points.append(max)



end = time.time()
print(end - start)


def my_atan(x, y):
    return np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
           -0.25*(2+np.sign(x))*np.sign(y))\
           -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))


# sort corner points
if len(corner_points) != 4:
    print('error: invalid corners')
    print(f'{len(corner_points)}')
    exit()

x_center = (corner_points[0][0] + corner_points[1][0] + corner_points[2][0] + corner_points[3][0]) * 0.25
y_center = (corner_points[0][1] + corner_points[1][1] + corner_points[2][1] + corner_points[3][1]) * 0.25

sorted_corners = sorted(corner_points, key=lambda point: my_atan(point[0] - x_center, point[1] - y_center))
for (idx, p) in enumerate(sorted_corners):
    print(f"{p[0]} | {p[1]} : {my_atan(p[0] - x_center, p[1] - y_center)}")
    if idx == 0:
        cv2.circle(img, (int((p[0]+0.5)/scalef), int((p[1]+0.5)/ scalef)), 3, color=(255,0,0), thickness=5)

    if idx == 1:
        cv2.circle(img, (int((p[0]+0.5)/scalef), int((p[1]+0.5)/ scalef)), 3, color=(0,255,0), thickness=5)

    if idx == 2:
        cv2.circle(img, (int((p[0]+0.5)/scalef), int((p[1]+0.5)/ scalef)), 3, color=(0,0,255), thickness=5)

DEBUG_OUTPUT and cv2.imwrite(os.path.join(DEBUG_OUTPUT, 'corner_points.jpg'), img)



# transform

# Locate points of the documents or object which you want to transform 

# height, width = downsampled.shape
pts2 = np.float32([
    [512-1, 512-1],
    [0, 512-1],
    [0, 0], 
    [512-1, 0],
    ]) 
pts1 = np.float32([
    *map(lambda p: [p[0] / scalef, p[1] / scalef], sorted_corners)
]) 
      
# Apply Perspective Transform Algorithm 
matrix = cv2.getPerspectiveTransform(pts1, pts2) 

# cv2.imwrite(os.path.join(output_dir, 'hough.jpg'), downsampled)

output = cv2.warpPerspective(original, matrix, (512, 512))

cv2.imwrite(os.path.join(output_dir, 'done.jpg'), output)



