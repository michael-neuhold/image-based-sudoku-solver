from math import sqrt
import cv2
import numpy as np
import os

# read image
dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, '..', 'images', 'input')
output_dir = os.path.join(dirname, '..', 'images', 'output')

img_path = os.path.join(input_dir, 'sudoku_001.jpeg')
img = cv2.imread(img_path)

height, width, _ = img.shape
fy = 512
fx = int(round((fy/height) * width))

# downsample
# img = cv2.resize(img, # original image
#                  (fx, fy), # set fx and fy, not the final size
#                  interpolation=cv2.INTER_CUBIC)
# 
# cv2.imwrite(os.path.join(output_dir, 'downsampled.jpg'), img)  

# blur
blurred = cv2.GaussianBlur(img, (17, 17), 0)

# loop over the image, pixel by pixel
sum = 0
for y in range(0, height):
    for x in range(0, width):
        # threshold the pixel
        color = blurred[y, x] * (1/255)
        r, g, b = color
        d0 = 1-abs(r-b)
        d1 = 1-abs(b-g)
        d2 = 1-abs(g-r)
        val = int((1-pow(d0*d1*d2, 1/3)) * 255)
        sum += val
        blurred[y, x] = [val, val, val]
avg = sum / (width*height)

for y in range(0, height):
    for x in range(0, width):
        val, _, _ =  blurred[y, x]
        val -= avg
        val = max(0, val)
        blurred[y, x] = [val, val, val]

cv2.imwrite(os.path.join(output_dir, 'gaussian.jpg'), blurred)  


# apply some filters
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 90, 150, apertureSize=5)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=3)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.erode(edges, kernel, iterations=4)

# save filtered image (canny.jpg)
cv2.imwrite(os.path.join(output_dir, 'canny.jpg'), edges)

# apply HoughLines
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/360, threshold=410)


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
