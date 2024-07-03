import cv2
import numpy as np
import cv2 as cv


def compute_gradient(img):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=7)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=7)
    return [sobelx, sobely]


def compute_magnitude(sobel):
    return np.sqrt(sobel[0] ** 2 + sobel[1] ** 2)


def compute_orientation(sobel):
    return np.arctan2(sobel[1], sobel[0]) * (180 / np.pi) % 180


def create_histogram_of_oriented_gradients(magnitude, orientation, cell_size=(8, 8), bins=9):
    cell_rows = magnitude.shape[0] // cell_size[0]
    cell_cols = magnitude.shape[1] // cell_size[1]
    histogram = np.zeros((cell_rows, cell_cols, bins))

    bin_size = 180 // bins

    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_magnitude = magnitude[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
            cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                               j * cell_size[1]:(j + 1) * cell_size[1]]

            for k in range(cell_magnitude.shape[0]):
                for l in range(cell_magnitude.shape[1]):
                    mag = cell_magnitude[k, l]
                    angle = cell_orientation[k, l]
                    bin_idx = int(angle // bin_size)
                    p = angle // bin_size
                    r = p - bin_idx
                    next_bin = bin_idx + 1 if bin_idx + 1 <= 8 else 0
                    histogram[i, j, next_bin] += r * mag
                    histogram[i, j, bin_idx] += (1 - r) * mag

    return histogram


def draw_hog_cells(histogram, cell_size=(8, 8), bins=9):
    cell_rows, cell_cols, _ = histogram.shape
    img_hog = np.full((cell_rows * cell_size[0], cell_cols * cell_size[1]), 255, dtype=np.uint8)

    max_mag = np.max(histogram)
    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_hist = histogram[i, j, :]
            max_bin = np.argmax(cell_hist)
            angle = (max_bin * (180 / bins) + (180 / bins) / 2) - 90
            magnitude = cell_hist[max_bin] / max_mag * cell_size[0]
            angle_rad = np.deg2rad(angle)

            x1 = int(j * cell_size[1] + cell_size[1] / 2 + magnitude * np.cos(angle_rad))
            y1 = int(i * cell_size[0] + cell_size[0] / 2 - magnitude * np.sin(angle_rad))
            x2 = int(j * cell_size[1] + cell_size[1] / 2 - magnitude * np.cos(angle_rad))
            y2 = int(i * cell_size[0] + cell_size[0] / 2 + magnitude * np.sin(angle_rad))

            cv.line(img_hog, (x1, y1), (x2, y2), 0, 1)

    return img_hog


def compute_phog(img, level=3, cell_size=(8, 8), bins=9):
    phog_descriptor = []

    for l in range(level):
        divisions = 2 ** l
        division_size_x = img.shape[1] // divisions
        division_size_y = img.shape[0] // divisions

        for i in range(divisions):
            for j in range(divisions):
                region = img[j * division_size_y:(j + 1) * division_size_y,
                         i * division_size_x:(i + 1) * division_size_x]

                region_sobel = compute_gradient(region)

                region_mag = compute_magnitude(region_sobel)
                region_orientation = compute_orientation(region_sobel)

                region_histogram = create_histogram_of_oriented_gradients(region_mag, region_orientation, cell_size,
                                                                          bins)

                phog_descriptor.append(region_histogram.flatten())

    return np.concatenate(phog_descriptor)

# if __name__ == "__main__":
#   img = cv2.imread("../my-signatures/1-fake.jpeg", cv2.IMREAD_GRAYSCALE)
#   _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#   compute_phog(img)
# sobel = compute_gradient(img)
# mag = compute_magnitude(sobel)
# orientation = compute_orientation(sobel)
# histogram = create_histogram_of_oriented_gradients(mag, orientation)
# hog_image = draw_hog_cells(img, histogram)
#
# cv.imshow("histogram", hog_image)
# cv.waitKey(0)
