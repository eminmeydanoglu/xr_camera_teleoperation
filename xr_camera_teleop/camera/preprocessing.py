"""Equivalent accelerated Gaussian for official WiLoR's full RGB crop prep."""
import cv2
import numpy as np
from skimage.filters import gaussian as reference_gaussian


def gaussian_rgb(image, sigma, *, channel_axis=None, preserve_range=False,
                 mode="nearest", truncate=4., **kwargs):
    if (channel_axis != 2 or not preserve_range or mode != "nearest" or kwargs
            or image.ndim != 3 or not np.isscalar(sigma) or image.dtype != np.uint8):
        return reference_gaussian(image, sigma=sigma, channel_axis=channel_axis,
            preserve_range=preserve_range, mode=mode, truncate=truncate, **kwargs)
    # scipy.ndimage's exact finite support and nearest boundary extension.
    # Use float64 to preserve the reference's numeric range and precision.
    radius = int(truncate * float(sigma) + .5)
    values = image.astype(np.float64)
    if radius == 0:
        return values
    return cv2.GaussianBlur(values, (2*radius+1, 2*radius+1),
        sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
