'''
Image crop detection by absolute patch localization.
Data perturbation logic for analyzing lens artefacts in neural network representations.
Basile Van Hoorick, Fall 2020.
'''

import cv2
import numpy as np
import PIL
import threading
from dataclasses import dataclass


@dataclass
class AberrationConfig:
    ''' Parameters for the lens aberration pipeline. '''

    # Percentage; positive = outward.
    red_chromatic_aberration: float = 0.0
    green_chromatic_aberration: float = 0.0
    blue_chromatic_aberration: float = 0.0

    # 0.0 = identity, 1.0 = maximal.
    vignetting_strength: float = 0.0

    # 1.0 = identity, more is exaggerated, less is reduced.
    color_saturation_factor: float = 1.0

    # Recommended use instead of color_saturation_factor = 0.0.
    grayscale_green: bool = False

    # k1 parameter in units of 1e-6.
    radial_distortion_k1: float = 0.0


def apply_chromatic_aberration(image, r_scale, g_scale, b_scale):
    '''
    Inserts artificial transverse chromatic aberration by linearly resizing
    the color channels with the specified relative factors (all should be >= 1.0).
    Final image has the same dimensions as the input.
    '''
    width, height = image.shape[1], image.shape[0]

    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood.
    result = image.copy()
    if r_scale != 1:
        fx = fy = r_scale
        mapMatrix = np.array([[fx, 0, -(fx-1)*width/2], [0, fy, -(fy-1)*height/2]])
        result[:, :, 0] = cv2.warpAffine(image[:, :, 0], mapMatrix,
                                         (width, height), flags=cv2.INTER_CUBIC)
    if g_scale != 1:
        fx = fy = g_scale
        mapMatrix = np.array([[fx, 0, -(fx-1)*width/2], [0, fy, -(fy-1)*height/2]])
        result[:, :, 1] = cv2.warpAffine(image[:, :, 1], mapMatrix,
                                         (width, height), flags=cv2.INTER_CUBIC)
    if b_scale != 1:
        fx = fy = b_scale
        mapMatrix = np.array([[fx, 0, -(fx-1)*width/2], [0, fy, -(fy-1)*height/2]])
        result[:, :, 2] = cv2.warpAffine(image[:, :, 2], mapMatrix,
                                         (width, height), flags=cv2.INTER_CUBIC)

    return result


def create_vignette_gain(width, height, strength):
    '''
    Creates a vignette gain map based on a six-degree polynomial [Rojas2015].
    '''
    a, b, c = np.float32(2.0625), np.float32(8.75), np.float32(0.0313)

    width, height, strength = np.float32(width), np.float32(height), np.float32(strength)
    radius = np.float32(np.sqrt(width * width + height * height, dtype=np.float32) / np.float32(2))

    x = np.float32(np.float32(np.arange(width) - width / 2.0) / radius)
    y = np.float32(np.float32(np.arange(height) - height / 2.0) / radius)
    x2, y2 = np.meshgrid(x, y, copy=False)
    r = np.sqrt(np.square(x2) + np.square(y2), dtype=np.float32)

    gain = np.float32(np.float32(1) +
                      a * np.power(r, np.float32(2.0), dtype=np.float32) +
                      b * np.power(r, np.float32(4.0), dtype=np.float32) +
                      c * np.power(r, np.float32(6.0), dtype=np.float32))
    gain = np.reciprocal(gain, dtype=np.float32)
    weighted_gain = np.float32(gain * strength + np.float32(1) - strength)

    final_gain = np.zeros((int(height), int(width), 3), dtype=np.float32)
    final_gain[:, :, 0] = weighted_gain
    final_gain[:, :, 1] = weighted_gain
    final_gain[:, :, 2] = weighted_gain
    return final_gain


def apply_vignette(image, vignette_gain):
    '''
    Inserts an artificial circular vignetting effect by multiplying with the given gain.
    '''
    if image.dtype.kind == 'f':
        image = np.float32(image)  # avoid 64 bit
    else:
        image = np.divide(image, 255.0, dtype=np.float32)
    image = np.float32(image * np.float32(vignette_gain))
    return image


def adjust_saturation(image, saturation_factor):
    '''
    Exaggerates or reduces saturation in a given numpy image (identity = 1, grayscale = 0).
    Note: this is faster to apply on a patch and does not need the full image.
    '''
    # print('1', image.min(), image.max(), image.dtype)
    if image.dtype.kind == 'f':
        image = (image * 255.0).astype(np.uint8)  # Convert from float32 to uint8.
    # print('2', image.min(), image.max(), image.dtype)
    image = PIL.Image.fromarray(image)
    image = PIL.ImageEnhance.Color(image).enhance(saturation_factor)
    image = np.array(image)
    # print('3', image.min(), image.max(), image.dtype)
#     if image.dtype.kind != 'f':
#         image = (image / 255.0).astype(np.float32)  # Ensure output is float32.
#     print('4', image.min(), image.max(), image.dtype)
    return image


def convert_grayscale(image):
    ''' Copies the green channel to red and blue channels in a given numpy image. '''
    # Ensure not read-only.
    if not(image.flags.writeable):
        image = image.copy()
    image[:, :, 0] = image[:, :, 1]
    image[:, :, 2] = image[:, :, 1]
    return image


def apply_lens_distortion(image, k1_e6):
    '''
    Applies barrel (positive) or pincushion (negative) radial distortion to the image.
    The parameter k1 is divided by 10^6 before applying, see formula at:
    https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    '''
    width, height = image.shape[1], image.shape[0]
    fx = 1
    fy = 1
    cameraMatrix = np.array([[fx, 0, (width-1)/2], [0, fy, (height-1)/2],
                             [0, 0, 1]]).astype(np.float32)
    k1 = k1_e6 * 1e-6
    k2 = 0
    p1 = 0
    p2 = 0
    alpha = 0
    distCoeffs = np.array([k1, k2, p1, p2])
    newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (width, height), alpha)
    image = cv2.undistort(image, cameraMatrix, distCoeffs, None, newCameraMatrix)
    return image


class ImageAberrationPipeline:
    '''
    Modifies every incoming image according to a fixed lens aberration profile.
    '''

    def __init__(self, config):
        self._config = config
        self._vignette_gains = dict()  # Maps resolution to mask.
        self._lock = threading.Lock()  # For vignetting updates.

    def process_image(self, image):
        '''
        Returns a distorted variant of the given clean image.
        '''
        width, height = image.shape[1], image.shape[0]
        full_res = (width, height)

        # Apply chromatic aberration if specified.
        r_scale = 1.0 + self._config.red_chromatic_aberration / 100.0
        g_scale = 1.0 + self._config.green_chromatic_aberration / 100.0
        b_scale = 1.0 + self._config.blue_chromatic_aberration / 100.0
        if r_scale != 1.0 or g_scale != 1.0 or b_scale != 1.0:
            image = apply_chromatic_aberration(image, r_scale, g_scale, b_scale)

        # Apply vignette artefact if specified.
        if self._config.vignetting_strength != 0.0:
            # Update dictionary if needed under a lock.
            with self._lock:
                if not(full_res in self._vignette_gains):
                    self._vignette_gains[full_res] = create_vignette_gain(
                        width, height, self._config.vignetting_strength)
            image = apply_vignette(image, self._vignette_gains[full_res])

        # Apply grayscale if specified.
        if self._config.grayscale_green:
            image = convert_grayscale(image)

        # Apply color saturation if specified.
        elif self._config.color_saturation_factor != 1.0:
            image = adjust_saturation(image, self._config.color_saturation_factor)
        
        # Apply radial lens barrel / pincushion distortion if specified.
        if self._config.radial_distortion_k1 != 0.0:
            image = apply_lens_distortion(image, self._config.radial_distortion_k1)

        return image
