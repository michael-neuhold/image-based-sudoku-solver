from math import pi, sqrt
from typing import List, Tuple
import numpy as np

def my_atan(x, y):
    return np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
        -0.25*(2+np.sign(x))*np.sign(y))\
        -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))


def filter_similar(lines, debug_output=None) -> List:
    if lines is None:
        print(f'error no lines found')
        return []
    elif len(lines) == 0 or len(lines) > 1000:
        print(f'error line-count: {len(lines)}')
        return []

    rho_threshold = 20
    theta_threshold = np.cos(16/180 * np.pi)

    # how many lines are similar to a given one
    similar_lines = [[] for _ in range(len(lines))]
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]

            diff_rad = theta_i - theta_j
            diff = abs(np.cos(diff_rad))

            if (abs(abs(rho_i) - abs(rho_j)) < rho_threshold and 
                diff > theta_threshold):
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

            if (abs(abs(rho_i) - abs(rho_j)) < rho_threshold and 
                diff > theta_threshold):
                # if it is similar and have not been disregarded yet then drop it now
                line_flags[indices[j]] = False

    debug_output and print('number of Hough lines:', len(lines))

    filtered_lines = []

    for i in range(len(lines)):  # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    debug_output and print('Number of filtered lines:', len(filtered_lines))

    return filtered_lines


def split_horizontal_vertical(lines_complete: List) -> Tuple[List, List]:
    horizontal_lines = []
    vertical_lines = []
    for line1 in lines_complete:
        rho0, theta0 = lines_complete[0][0]
        rho1, theta1 = line1[0]

        # calc difference
        # 1 -> same
        # 0 -> 90° difference
        diff = theta1 - theta0
        diff = abs(np.cos(diff))

        if diff < np.cos(45/180 * np.pi):
            horizontal_lines.append(line1)
        else:
            vertical_lines.append(line1)

    return horizontal_lines, vertical_lines


def calc_intersection(vline, hline) -> Tuple:
    vrho, vtheta = vline
    hrho, htheta = hline
    # x * cos(th) + y * sin(th) = rh 
    # x * cos(tv) + y * sin(tv) = rv

    #    ┌   ┐     ┌                  ┐ −1    ┌    ┐
    #    │ x │     │ cos(th)  sin(th) │       │ rh │
    #    │   │  =  │                  │   *   │    │
    #    │ y │     │ cos(tv)  sin(tv) │       │ rv │
    #    └   ┘     └                  ┘       └    ┘

    A = np.array([
            [ np.cos(htheta), np.sin(htheta) ],
            [ np.cos(vtheta), np.sin(vtheta) ]
        ])
    B = np.array([
            [ hrho ],
            [ vrho ]
        ])

    Ainv = np.linalg.inv(A) 
    X = Ainv @ B

    return (X[0][0], X[1][0])


def calc_xy_stddev(points) -> Tuple:
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
    
    # sqrt not needed because only 
    # comparison is evaluated
    stdevx = xdsum # sqrt(xdsum)
    stdevy = ydsum # sqrt(ydsum)

    return (stdevx, stdevy)


def get_minmax(points, stddevx, stddevy) -> Tuple:
    minp = None
    maxp = None
    if stddevx > stddevy:
        minp = min(points, key=lambda p: p[0])
        maxp = max(points, key=lambda p: p[0])
    else:
        minp = min(points, key=lambda p: p[1])
        maxp = max(points, key=lambda p: p[1])

    return (minp, maxp)


def calc_oriented_corners(horizontal_lines, vertical_lines) -> List:
    # get min/max intersection-points on horizontal lines
    edge_points = [ [] for _ in range(10) ] #...
    for hindex, hline in enumerate(horizontal_lines):
        points = []
        for vindex, vline in enumerate(vertical_lines):
            (x, y) = calc_intersection(vline[0], hline[0])
            points.append((x, y, hindex, vindex))
            
        stdevx, stdevy = calc_xy_stddev(points)
        minp, maxp = get_minmax(points, stdevx, stdevy)
        
        edge_points[minp[3]].append(minp)
        edge_points[maxp[3]].append(maxp)


    # filter points for vertical min/max
    corner_points = []
    for points in edge_points:
        if len(points) <= 0: continue

        stdevx, stdevy = calc_xy_stddev(points)
        minp, maxp = get_minmax(points, stdevx, stdevy)
        
        corner_points.append(minp)
        corner_points.append(maxp)


    if len(corner_points) != 4:
        print(f'error: invalid corner count ({len(corner_points)})')
        return []

    x_center = (corner_points[0][0] + corner_points[1][0] + corner_points[2][0] + corner_points[3][0]) * 0.25
    y_center = (corner_points[0][1] + corner_points[1][1] + corner_points[2][1] + corner_points[3][1]) * 0.25

    oriented_corners = sorted(corner_points, key=lambda point: my_atan(point[0] - x_center, point[1] - y_center))
    return oriented_corners