import cv2
import numpy as np
import os

# read image
img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) +
                 '/../images/sudoku1.png')

# apply some filters
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 90, 150, apertureSize=5)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
kernel = np.ones((5, 5), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)

# save filtered image (canny.jpg)
cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
            '/../images/canny.jpg', edges)

# apply HoughLines
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

if not lines.any():
    print('No lines were found')
    exit()

rho_threshold = 15
theta_threshold = 0.1

# how many lines are similar to a given one
similar_lines = {i: [] for i in range(len(lines))}
for i in range(len(lines)):
    for j in range(len(lines)):
        if i == j:
            continue

        rho_i, theta_i = lines[i][0]
        rho_j, theta_j = lines[j][0]
        if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
            similar_lines[i].append(j)

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
        if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
            # if it is similar and have not been disregarded yet then drop it now
            line_flags[indices[j]] = False

print('number of Hough lines:', len(lines))

filtered_lines = []

for i in range(len(lines)):  # filtering
    if line_flags[i]:
        filtered_lines.append(lines[i])

print('Number of filtered lines:', len(filtered_lines))

for line in filtered_lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# print image with highlighted, detected lines
cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
            '/../images/hough.jpg', img)
