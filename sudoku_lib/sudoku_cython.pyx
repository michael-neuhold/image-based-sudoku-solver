#!python
#cython: language_level=3

import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :] calc_component_bound(unsigned char[:, :] comp_img, unsigned char[:, :] bound_img):
    cdef int y, x, width, height
    height = comp_img.shape[0]
    width = comp_img.shape[1]

    # vertical lines
    for y in range(height):
        x = 0
        while (x < width and comp_img[y][x] == 0):
            x += 1

        if x < width:
            bound_img[y][x] = 255
            x = width-1
            while (x > 0 and comp_img[y][x] == 0):
                x -= 1

            if x > 0:
                bound_img[y][x] = 255

    # horizontal lines
    for x in range(width):
        y = 0
        while (y < height and comp_img[y][x] == 0):
            y += 1

        if y < height:
            bound_img[y][x] = 255
            y = height-1
            while (y > 0 and comp_img[y][x] == 0):
                y -= 1

            if y > 0:
                bound_img[y][x] = 255
    

    return bound_img