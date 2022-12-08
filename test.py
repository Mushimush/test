import cv2 as cv
import numpy as np
from constants import yiq_from_rgb, rgb_from_yiq, gaussian_kernel
from processing import pyrDown, pyrUp, reconstructLaplacianImage
from scipy.signal import butter


# Helper Methods


def rgb2yiq(rgb_image):
    image = rgb_image.astype(np.float32)
    return image @ yiq_from_rgb.T


cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture Frame by Frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exciting...")
        break

    # Our operations on the frame come here

    # Initialize Kernel
    kernel = gaussian_kernel

    # parameters
    level = 6
    alpha = 30
    lambda_cutoff = 16
    freq_range = [0.4, 3]
    attenuation = 3

    # Original Image
    original_image = frame.copy()

    # From RGB to YIQ
    image = rgb2yiq(frame)

    # Generation of Laplacian Pyramid
    laplacian_pyramid = []
    prev_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=prev_image, kernel=kernel)
        upsampled_image = pyrUp(image=downsampled_image,
                                kernel=kernel,
                                dst_shape=prev_image.shape[:2])
        laplacian_pyramid.append(prev_image - upsampled_image)
        prev_image = downsampled_image
    pyramids = np.asarray(laplacian_pyramid, dtype='object')
    # print(laplacian_pyramid)

    # Filtering Pyramid
    filtered_pyramids = np.zeros_like(pyramids)
    delta = lambda_cutoff / (8 * (1 + alpha))
    b_low, a_low = butter(1, freq_range[0], btype='low', output='ba', fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype='low', output='ba', fs=fps)

    lowpass = pyramids[0]
    highpass = pyramids[0]
    filtered_pyramids[0] = pyramids[0]

    for i in range(1, pyramids.shape[0]):
        lowpass = (-a_low[1] * lowpass
                   + b_low[0] * pyramids[i]
                   + b_low[1] * pyramids[i - 1]) / a_low[0]
        highpass = (-a_high[1] * highpass
                    + b_high[0] * pyramids[i]
                    + b_high[1] * pyramids[i - 1]) / a_high[0]

        filtered_pyramids[i] = highpass - lowpass

        for lvl in range(1, level - 1):
            (height, width, _) = filtered_pyramids[i, lvl].shape
            lambd = ((height ** 2) + (width ** 2)) ** 0.5
            new_alpha = (lambd / (8 * delta)) - 1

            filtered_pyramids[i, lvl] *= min(alpha, new_alpha)
            filtered_pyramids[i, lvl][:, :, 1:] *= attenuation

    video = np.zeros_like(original_image)
    for i in range(original_image.shape[0]):
        video[i] = reconstructLaplacianImage(
            image=original_image[i],
            pyramid=filtered_pyramids[i],
            kernel=kernel
        )

    cv.imshow('test', video[i])
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
