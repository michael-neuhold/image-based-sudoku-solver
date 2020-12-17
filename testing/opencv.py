from math import sqrt
import cv2
import numpy as np
import os

# read image
dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, '..', 'images', 'input')
output_dir = os.path.join(dirname, '..', 'images', 'output')


def downsample(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, _ = img.shape
    fy = 512
    fx = int( round((fy/height) * width) )

    # downsample
    downsampled = cv2.resize(gray, # original image
                    (fx, fy), # set fx and fy, not the final size
                    interpolation=cv2.INTER_CUBIC)
    return downsampled

def get_edges(img):
    # cv2.imwrite(os.path.join(output_dir, 'downsampled.jpg'), downsampled)  

    # blur
    blurred = cv2.medianBlur(img, ksize=3)
    # cv2.imwrite(os.path.join(output_dir, 'median.jpg'), blurred)  

    # apply some filters
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    sobel_x = cv2.multiply(sobel_x, sobel_x)
    sobel_y = cv2.multiply(sobel_y, sobel_y)
    sobel = cv2.add(sobel_x, sobel_y)
    sobel = cv2.sqrt(sobel)
    sobel = sobel / sobel.max() * 255

    _, sobel = cv2.threshold(sobel, 50, 255, type=cv2.THRESH_BINARY)

    # cv2.imwrite(os.path.join(output_dir, 'sobel.jpg'), sobel)  

    sobel = sobel.astype('uint8')

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(sobel, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # cv2.imwrite(os.path.join(output_dir, 'eroded.jpg'), eroded)

    return eroded

def get_biggest_connected_component(img):
    # calculate segments
    component_count, component_img, stats, centroids = ( 
        cv2.connectedComponentsWithStats(edge_img, connectivity=8))

    id_area_pairs = list(
        map(
            lambda x: (x[0], x[1][cv2.CC_STAT_AREA]),
            enumerate(stats)
           )
        )

    stats = sorted(id_area_pairs, key=lambda stat: stat[1], reverse=True)
    
    component_id = stats[1][0]
    print(component_id)
    
    # select biggest component
    component = cv2.inRange(component_img, component_id, component_id)
    # cv2.imwrite(os.path.join(output_dir, 'components.jpg'), component_img)

    return component



img = cv2.imread(
        os.path.join(input_dir, 'sudoku_001.jpeg'))

downsampled = downsample(img)
edge_img  = get_edges(downsampled)
component = get_biggest_connected_component(edge_img)

cv2.imwrite(os.path.join(output_dir, 'component.jpg'), component)


# apply HoughLines
lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=300)


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


for line in horizontal_lines:
    rho, theta = line[0]
    print(theta)


rho, theta = horizontal_lines[0][0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(downsampled, (x1, y1), (x2, y2), (0, 0, 255), 2)
rho, theta = vertical_lines[0][0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(downsampled, (x1, y1), (x2, y2), (0, 0, 255), 2)

# print image with highlighted, detected lines

for hline in list(horizontal_lines[0]):
    for vline in list(vertical_lines[0]):
        vrho, vtheta = vline[0]
        hrho, htheta = hline[0]

        # (vrho - y * sin(vtheta)) / cos(vtheta) = x
        # hrho = x * cos(htheta) + y * sin(htheta)

        y = ((vrho* np.cos(htheta) / np.cos(vtheta) - hrho) / 
             (np.sin(vtheta) * np.cos(htheta) / np.cos(vtheta) + np.sin(htheta)))

        x = (vrho - y * np.sin(vtheta)) / np.cos(vtheta)

        print(f'x: {x} | y: {y}')
        h, w = downsampled.shape
        cv2.circle(downsampled, (int(h - y), int(w -x)), 10, color=255)

cv2.imwrite(os.path.join(output_dir, 'hough.jpg'), downsampled)


