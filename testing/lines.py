from typing import List, Tuple
import numpy as np

def filter_similar(lines, debug_output=None) -> List:
    if lines is None:
        print('no lines found')
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


def split_horizantal_vertical(lines_complete: List) -> Tuple[List, List]:
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
