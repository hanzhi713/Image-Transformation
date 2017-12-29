import numpy as np
import cv2, time
import matplotlib.pyplot as plt


def transform(x, y, orgX, orgY):
    c = complex(x - orgX, y - orgY)
    return c ** 1.2


const = np.array([256, 256, 256], np.int16)


# convert the sparse matrix dictionary (mapping (x, y) to (b, g, r)) to a numpy three dimensional array
def toMatrix(newDict):
    global const
    arrs = newDict.keys()
    xRange = max(arrs, key=lambda x: x[0])[0] - min(arrs, key=lambda x: x[0])[0]
    yRange = max(arrs, key=lambda x: x[1])[1] - min(arrs, key=lambda x: x[1])[1]
    shiftX = xRange // 2
    shiftY = yRange // 2
    imgArr = np.zeros((yRange, xRange, 3), np.int16)
    for x in range(xRange):
        for y in range(yRange):
            imgArr[y, x, :] = np.array(newDict.get((x - shiftX, y - shiftY), [255, 255, 255]), np.int16)
    return const - imgArr


def bgrTorgb(img):
    img_rgb = np.zeros(img.shape, img.dtype)
    img_rgb[:, :, 0] = img[:, :, 2]
    img_rgb[:, :, 1] = img[:, :, 1]
    img_rgb[:, :, 2] = img[:, :, 0]
    return img_rgb


# display the image of the three dimensional image array
def show(ori, img):
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(bgrTorgb(ori))
    plt.subplot(122)
    plt.title('Destination Image')
    plt.imshow(bgrTorgb(img))
    plt.show()


# interpolate the pixels with a matrix of size (size*size)
def avPixels(newImg, m, n, bgr, size, c):
    a = round(m)
    b = round(n)
    for i in range(-c, size - c):
        for j in range(-c, size - c):
            (x, y) = (a + i, b + j)
            if newImg.get((x, y)) is None:
                newImg[(x, y)] = bgr


def main():
    t = time.clock()
    img = cv2.imread("pics/5.png")
    height, width = img.shape[0:2]
    orgX, orgY = (width // 2, height // 2)

    # the kernel size
    kernel = 7
    c = kernel // 2
    newImg = {}
    for x in range(width):
        for y in range(height):
            xy = transform(x, y, orgX, orgY)
            avPixels(newImg, xy.real, xy.imag, img[y, x, :], kernel, c)
    imgArr = toMatrix(newImg)
    print(time.clock() - t)
    show(img, imgArr)


if __name__ == "__main__":
    main()
