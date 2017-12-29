import numpy as np
import cv2, time, multiprocessing
import matplotlib.pyplot as plt


def transform(x, y, orgX, orgY):
    c = complex(x - orgX, y - orgY)
    return c ** 1.2


const = np.array([256, 256, 256], np.int16)
def toMatrix(newDict):
    global const
    arrs = newDict.keys()
    xRange = max(arrs, key=lambda x: x[0])[0] - min(arrs, key=lambda x: x[0])[0]
    yRange = max(arrs, key=lambda x: x[1])[1] - min(arrs, key=lambda x: x[1])[1]
    print("Rendering image of size {}x{}...".format(xRange, yRange))
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


def show(ori, img):
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(bgrTorgb(ori))
    plt.subplot(122)
    plt.title('Destination Image')
    plt.imshow(bgrTorgb(img))
    plt.show()


def avPixels(newImg, m, n, bgr, c):
    a = round(m)
    b = round(n)
    for i in range(a - c, a + c):
        for j in range(b - c, b + c):
            if newImg.get((i, j)) is None:
                newImg[(i, j)] = bgr


def calculateSparseArray(img, wStart, wEnd, h, orgX, orgY, kernel):
    c = kernel // 2
    newImg = {}
    for x in range(wStart, wEnd):
        for y in range(h):
            xy = transform(x, y, orgX, orgY)
            avPixels(newImg, xy.real, xy.imag, img[y, x, :], c)
    return newImg

def main():
    img = cv2.imread("pics/5.png")
    t = time.clock()
    height, width = img.shape[0:2]
    orgX, orgY = (width // 2, height // 2)
    kernel = 7
    threads = 6
    wPart = width // threads
    results = []
    pool = multiprocessing.Pool(processes=threads)
    for i in range(threads - 1):
        results.append(
            pool.apply_async(calculateSparseArray, (img, wPart * i, wPart * (i + 1), height, orgX, orgY, kernel,)))
    results.append(
        pool.apply_async(calculateSparseArray, (img, wPart * (threads - 1), width, height, orgX, orgY, kernel,)))
    pool.close()
    pool.join()
    print('It takes {}s to calculate the sparse matrices, with kernel of size {}x{}, using {} threads'
          .format(round(time.clock() - t, 2), kernel, kernel, threads))
    t = time.clock()
    d1 = results[0].get()
    for i in range(1, len(results)):
        d1.update(results[i].get())
    print('It takes {}s to merge the sparse matrices'.format(round(time.clock() - t, 2)))
    t = time.clock()
    imgArr = toMatrix(d1)
    print('It takes {}s to convert sparse matrices to a complete numpy three dimensional array'.format(
        round(time.clock() - t, 2)))
    show(img, imgArr)


if __name__ == "__main__":
    main()
