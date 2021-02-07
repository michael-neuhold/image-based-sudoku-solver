from math import inf, pi, sqrt
from typing import List, Tuple
import numpy as np


def my_atan(x, y):
    return np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
        -0.25*(2+np.sign(x))*np.sign(y))\
        -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))


##def filter_similar(lines, debug_output=None) -> List:
##    if lines is None:
##        print(f'error: no lines found')
##        return []
##    elif len(lines) == 0 or len(lines) > 1000:
##        print(f'error: line-count = {len(lines)}')
##        return []
##
##    rho_threshold = 20
##    theta_threshold = np.cos(16/180 * np.pi)
##
##    # how many lines are similar to a given one
##    similar_lines = [[] for _ in range(len(lines))]
##    for i in range(len(lines) - 1):
##        for j in range(i + 1, len(lines)):
##            rho_i, theta_i = lines[i][0]
##            rho_j, theta_j = lines[j][0]
##
##            diff_rad = theta_i - theta_j
##            diff = abs(np.cos(diff_rad))
##
##            if (abs(abs(rho_i) - abs(rho_j)) < rho_threshold and 
##                diff > theta_threshold):
##                similar_lines[i].append(j)
##                similar_lines[j].append(i)
##
##    # ordering the INDECES of the lines by how many are similar to them
##    indices = [i for i in range(len(lines))]
##    indices.sort(key=lambda x: len(similar_lines[x]))
##
##    # line flags is the base for the filtering
##    line_flags = len(lines)*[True]
##    for i in range(len(lines) - 1):
##        # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
##        if not line_flags[indices[i]]:
##            continue
##
##        # we are only considering those elements that had less similar line
##        for j in range(i + 1, len(lines)):
##            # and only if we have not disregarded them already
##            if not line_flags[indices[j]]:
##                continue
##
##            rho_i, theta_i = lines[indices[i]][0]
##            rho_j, theta_j = lines[indices[j]][0]
##
##            diff_rad = theta_i - theta_j
##            diff = abs(np.cos(diff_rad))
##
##            if (abs(abs(rho_i) - abs(rho_j)) < rho_threshold and 
##                diff > theta_threshold):
##                # if it is similar and have not been disregarded yet then drop it now
##                line_flags[indices[j]] = False
##
##    debug_output and print('number of Hough lines:', len(lines))
##
##    filtered_lines = []
##
##    for i in range(len(lines)):  # filtering
##        if line_flags[i]:
##            filtered_lines.append(lines[i])
##
##    debug_output and print('Number of filtered lines:', len(filtered_lines))
##
##    return filtered_lines


def filter_similar_new(lines, width, height, debug_output=None) -> List:
    if lines is None:
        print(f'error: no lines found')
        return []
    elif len(lines) == 0 or len(lines) > 5000:
        print(f'error: line-count = {len(lines)}')
        return []

    theta_threshold = np.cos(16/180 * np.pi)

    # group similar lines together
    similar_line_collection = []
    used = [False for _ in range(len(lines))]
    intersects_with = [-1 for _ in range(len(lines))]
    line_collection_id_for_parent_line = [-1 for _ in range(len(lines))]
    for i in range(len(lines) - 1):
        similar_lines = []
        for j in range(i + 1, len(lines)):
            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]

            diff_rad = theta_i - theta_j
            diff = abs(np.cos(diff_rad))

            if diff > theta_threshold:
                intersection = calc_intersection(lines[i][0], lines[j][0])
                if intersection:
                    x, y = intersection
                    if (x > 0 and x < width and
                        y > 0 and y < height):
                        if not used[j]:
                            similar_lines.append(lines[j])
                            used[j] = True
                            intersects_with[j] = i
                        else:
                            parent_line = intersects_with[j]
                            line_collection_id = line_collection_id_for_parent_line[parent_line]
                            similar_line_collection[line_collection_id].append(lines[j])
                            used[j] = True
                            intersects_with[j] = parent_line

        if not used[i]:
            similar_lines.append(lines[i])
            used[i] = True
        
        if len(similar_lines) > 0:
            line_collection_id_for_parent_line[i] = len(similar_line_collection)
            similar_line_collection.append(similar_lines)


    # process similar lines
    filtered_lines = []
    for similar_lines in similar_line_collection:
        avg_theta = 0
        avg_rho = 0
        for line in similar_lines:
            rho, theta = line[0]
            if rho < 0:
                rho = abs(rho)
                theta = theta - pi
            avg_rho += rho
            avg_theta += theta

        avg_theta /= len(similar_lines)
        avg_rho /= len(similar_lines)

        # sudoku.render_lines(img, similar_lines, scalef, (200,200,200))
        # sudoku.render_lines(img, [[ (avg_rho, avg_theta) ]], scalef, (0, 255, 0))
        filtered_lines.append([ (avg_rho, avg_theta) ])

    return filtered_lines


# TODO:
# could be improved => lines that are neither
# vertical nor horizontal are not filtered
# idea: * 'buckets' of similar lines (theta)
#       * 'dissimilar' line creates new bucket
#       * the two largest buckets are assumed to be the 
#         horizontal/vertical lines
def filter_outliers(lines, debug_output=None) -> List:
    if (lines is None) or len(lines) == 0:
        return []

    theta_threshold = np.cos(16/180 * np.pi)

    # group similar lines together
    similar_line_collection = []
    used = [False for _ in range(len(lines))]
    for i in range(len(lines) - 1):
        similar_lines = []
        for j in range(i + 1, len(lines)):
            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]

            diff_rad = theta_i - theta_j
            diff = abs(np.cos(diff_rad))

            if (not used[j]) and (diff > theta_threshold):
                similar_lines.append(lines[j])
                used[j]= True
        
        if not used[i]:
            similar_lines.append(lines[i])
            used[i] = True
        
        if len(similar_lines) > 0:
            similar_line_collection.append(similar_lines)

    if len(similar_line_collection) != 2:
        print(f'similar: {len(similar_line_collection)}')
        
    if (len(similar_line_collection) == 0 or
        len(similar_line_collection) == 1):
        return []
    
    sorted_s_line_collection = sorted(similar_line_collection, key=lambda s_lines: len(s_lines), reverse=True)
    filtered_lines = [ *sorted_s_line_collection[0], *sorted_s_line_collection[1] ]

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

    try:
        Ainv = np.linalg.inv(A) 
    except:
        return None

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
    edge_points = [ [] for _ in range(100) ] #...
    for hindex, hline in enumerate(horizontal_lines):
        points = []
        for vindex, vline in enumerate(vertical_lines):
            intersection = calc_intersection(vline[0], hline[0])
            if intersection:
                x, y = intersection
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