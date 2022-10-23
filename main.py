import math

import numpy as np
import tqdm as tqdm

from mnist import load_images, load_tags
from sklearn import svm


def hog(image):
    """Compute the histogram of oriented gradients of an image."""
    width, height = image.shape

    # Compute the gradient of the image
    gx = np.zeros(image.shape, dtype=np.float32)
    gy = np.zeros(image.shape, dtype=np.float32)

    for i in range(0, width):
        for j in range(0, height):
            if i == 0:
                gx[i, j] = image[i + 1, j] - 0
            elif i == width - 1:
                gx[i, j] = 0 - image[i - 1, j]
            else:
                gx[i, j] = float(image[i + 1, j]) - float(image[i - 1, j])

            if j == 0:
                gy[i, j] = image[i, j + 1] - 0
            elif j == height - 1:
                gy[i, j] = 0 - image[i, j - 1]
            else:
                gy[i, j] = float(image[i, j + 1]) - float(image[i, j - 1])

    gx /= 255
    gy /= 255

    # Compute the magnitude and orientation of the gradient
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.zeros(gx.shape)
    angle[gx != 0] = np.abs(np.arctan(gy[gx != 0] / gx[gx != 0]))
    step = math.pi / 9

    # Compute the histogram of oriented gradients
    histogram = np.zeros((width // 8, height // 8, 9))
    for i in range(width // 8):
        for j in range(height // 8):
            for offset_x in range(8):
                for offset_y in range(8):
                    b = math.floor(angle[i * 8 + offset_x, j * 8 + offset_y] / step - 0.5)
                    bin_weight = step * (j + 0.5)
                    next_bin_weight = (step * (j + 1.5) - angle[i * 8 + offset_x, j * 8 + offset_y]) / step

                    histogram[i, j, b] += magnitude[i * 8 + offset_x, j * 8 + offset_y] * bin_weight
                    histogram[i, j, (b + 1) % 9] += magnitude[i * 8 + offset_x, j * 8 + offset_y] * next_bin_weight

    # normalization and feature vector
    features = np.zeros(((width // 8 - 1) * (height // 8 - 1) * 36))
    for i in range(width // 8 - 1):
        for j in range(height // 8 - 1):
            window = histogram[i:i + 2, j:j + 2, :]
            norm = np.linalg.norm(window)
            window = window / (norm + 1e-6)
            start = (i * (width // 8 - 1) + j) * 36
            end = start + 36
            features[start:end] = window.flatten()

    return features


if __name__ == "__main__":
    images = load_images("train-images.idx3-ubyte")
    tags = load_tags("train-labels.idx1-ubyte")

    X = []
    for img in tqdm.tqdm(images):
        X.append(hog(img))

    clf = svm.SVC()
    clf.fit(X, tags)

    test_images = load_images("t10k-images.idx3-ubyte")
    test_tags = load_tags("t10k-labels.idx1-ubyte")
    predictions = np.zeros(test_tags.shape)
    for i, x in tqdm.tqdm(enumerate(test_images)):
        predictions[i] = clf.predict([hog(x)])

    print(np.sum(predictions == test_tags) / len(test_tags))
