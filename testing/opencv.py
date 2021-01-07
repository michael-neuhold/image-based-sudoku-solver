from math import pi, sqrt
import cv2
import numpy as np
import os

# read image
dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, '..', 'images', 'input')
output_dir = os.path.join(dirname, '..', 'images', 'output')


def downsample(img, fy):
    height, width = img.shape
    fx = int( round((fy/height) * width) )

    # downsample
    downsampled = cv2.resize(img, # original image
                    (fx, fy), # set fx and fy, not the final size
                    interpolation=cv2.INTER_CUBIC)
    return downsampled, (fy / height)

def get_edges(img):
    canny = cv2.Canny(img, 120, 400)

    kernel = np.array([
                        [ 0, 1, 1, 1, 0 ],
                        [ 1, 1, 1, 1, 1 ],
                        [ 1, 1, 1, 1, 1 ],
                        [ 1, 1, 1, 1, 1 ],
                        [ 0, 1, 1, 1, 0 ]
                     ], dtype=np.uint8)
    kernel3 = np.array([
                        [ 1, 1, 1 ],
                        [ 1, 1, 1 ],
                        [ 1, 1, 1 ],
                     ], dtype=np.uint8)
    
    # join edges => every line consists of 2 edges
    result = cv2.dilate(canny, kernel, iterations=1)

    # remove thin edges
    result = cv2.erode(result, kernel, iterations=1)
    result = cv2.erode(result, kernel3, iterations=1)

    # regrow remaining edges
    #result = cv2.dilate(canny, kernel3, iterations=1)

    return result

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
        os.path.join(input_dir, 'sudoku_005.jpg'))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
downsampled, scalef = downsample(gray, 480)
edge_img  = get_edges(downsampled)
cv2.imwrite(os.path.join(output_dir, 'component.jpg'), edge_img)
component = get_biggest_connected_component(edge_img)


exit()

# apply HoughLines
lines = cv2.HoughLines(component, rho=1, theta=np.pi/360, threshold=220)


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
    cv2.line(downsampled, (x1, y1), (x2, y2), (0, 0, 255), 2)

# print image with highlighted, detected lines
edge_points = { 0: [], 1:[], 2: [], 3: [], 4: [] } #...

for hindex, hline in enumerate(horizontal_lines):
    points = []
    for vindex, vline in enumerate(vertical_lines):
        vrho, vtheta = vline[0]
        hrho, htheta = hline[0]

        # rh = x * cos(th) + y * sin(th)
        # rv = x * cos(tv) + y * sin(tv)
       
        # (rh - y * sin(th)) / cos(th) = x
        # rv = x * cos(tv) + y * sin(tv)

        y = (vrho - (hrho * np.cos(vtheta) / np.cos(htheta))) / (np.sin(vtheta) - np.sin(htheta) * np.cos(vtheta) / np.cos(htheta))
        x = (hrho - y * np.sin(htheta)) / np.cos(htheta)

        # print(f'x: {x} | y: {y}')
        # h, w = downsampled.shape
        # cv2.circle(downsampled, (int(x), int(y)), 1, color=255, thickness=3)
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

    # print('min: ', min)
    # print('max: ', max)
    edge_points[min[3]].append(min)
    edge_points[max[3]].append(max)
    # cv2.circle(downsampled, (int(min[0]), int(min[1])), 1, color=255, thickness=3)
    # cv2.circle(downsampled, (int(max[0]), int(max[1])), 1, color=255, thickness=3)

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

    cv2.circle(downsampled, (int(min[0]), int(min[1])), 1, color=255, thickness=3)
    cv2.circle(downsampled, (int(max[0]), int(max[1])), 1, color=255, thickness=3)
    corner_points.append(min)
    corner_points.append(max)

# transform

# Locate points of the documents or object which you want to transform 

height, width = downsampled.shape
pts2 = np.float32([
    [0, 512-1],
    [512-1, 512-1],
    [0, 0], 
    [512-1, 0],
    ]) 
pts1 = np.float32([
    *map(lambda p: [p[0] / scalef, p[1] / scalef], corner_points)
]) 
      
# Apply Perspective Transform Algorithm 
matrix = cv2.getPerspectiveTransform(pts1, pts2) 

cv2.imwrite(os.path.join(output_dir, 'hough.jpg'), downsampled)

output = cv2.warpPerspective(img, matrix, (512, 512))

cv2.imwrite(os.path.join(output_dir, 'done.jpg'), output)



