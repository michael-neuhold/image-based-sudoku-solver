from math import sqrt
import cv2
import numpy as np
import os

# read image
dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, '..', 'images', 'input')
output_dir = os.path.join(dirname, '..', 'images', 'output')


img = cv2.imread(
            os.path.join(input_dir, 'sudoku_001.jpeg')
        )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

float_gray = gray.astype(np.float32) / 255.0

blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
num = float_gray - blur

blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=4, sigmaY=4)
den = cv2.pow(blur, 0.5)

gray = num / den

cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

cv2.imwrite(
        os.path.join(output_dir, 'norm.jpeg'),
        gray * 255
    )



exit()


img_path = os.path.join(input_dir, 'sudoku_001.jpeg')
img = cv2.imread(img_path)

gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cv2.imwrite(os.path.join(output_dir, 'adapt.jpg'), gray)  




converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

saturation = cv2.split(converted)[1]
value =  255 - cv2.split(converted)[2]

result = 255 - cv2.multiply(saturation, value, scale=1/255)
cv2.imwrite(os.path.join(output_dir, 'a.jpg'), saturation)  
cv2.imwrite(os.path.join(output_dir, 'b.jpg'), value)  
cv2.imwrite(os.path.join(output_dir, 'test.jpg'), result)  


exit()

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
blurred = cv2.GaussianBlur(img, (21, 21), 0)
cv2.imwrite(os.path.join(output_dir, 'gaussian.jpg'), blurred)  


# apply some filters
r, g, b = cv2.split(blurred)
edges0 = cv2.Canny(r, 300, 900, apertureSize=5, L2gradient=True)
edges1 = cv2.Canny(g, 300, 900, apertureSize=5)
edges2 = cv2.Canny(b, 300, 900, apertureSize=5)


# save filtered image (canny.jpg)
cv2.imwrite(os.path.join(output_dir, 'canny0.jpg'), edges0)
cv2.imwrite(os.path.join(output_dir, 'canny1.jpg'), edges1)
cv2.imwrite(os.path.join(output_dir, 'canny2.jpg'), edges2)

result = cv2.multiply(edges0, edges1)
result = cv2.multiply(result, edges2)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(result, kernel, iterations=4)
edges = cv2.erode(edges, kernel, iterations=4)
cv2.imwrite(os.path.join(output_dir, 'mul.jpg'), edges)


# apply HoughLines
lines = cv2.HoughLines(edges0, rho=1, theta=np.pi/360, threshold=200)


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
