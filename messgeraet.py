import argparse
from skimage import io, draw, feature, transform

import numpy as np
from skimage.util import img_as_ubyte

from dataclasses import dataclass

@dataclass
class Settings:
    threshold: int
    image_path: str
    output_step: str
    reference_object: int

def log(msg):
    print("Debug: " + msg)

def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Image pipeline configuration via CLI arguments"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=125,
        help="Threshold value for binarization (default: 125)"
    )
    parser.add_argument(
        "--image-path", "-i",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--output-step", "-s",
        type=str,
        choices=["last", "step1", "step2", "step3"], # TODO: Replace with actual steps
        default="last",
        help="Pipeline step for which to save the output image (default: last)"
    )
    parser.add_argument(
        "--reference-object", "-r",
        type=int,
        choices=[2, 1, 50],
        required=True,
        help="Reference object for scale (2€, 1€ or 50 cent)"
    )

    args = parser.parse_args()
    return Settings(
        threshold=args.threshold,
        image_path=args.image_path,
        output_step=args.output_step,
        reference_object=args.reference_object
    )

def save_current_pipeline_state(image, image_name):
    image_ubyte = img_as_ubyte(image)
    io.imsave(f'{image_name}.png', image_ubyte)
    log(f"Stored {image_name}.png")

def rgb2gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)

def detect_circle(binary_image):
    edges = feature.canny(binary_image)
    larger_edge = max(binary_image.shape[0], binary_image.shape[1])
    min_size = larger_edge * 0.25
    max_size = larger_edge * 0.5
    hough_radii = np.arange(min_size, max_size, 1)
    hough_res = transform.hough_circle(edges, hough_radii)

    accumulators, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=3
    )
    # ycirc, xcirc = draw.circle_perimeter_aa(cy[0], cx[0], radii[0], shape=binary_image.shape)
    # for y in ycirc:
    #     for x in xcirc:
    #         binary_image[y, x] = 0.5

    log(f"Computed center and radius: x={cx} y={cy} rad={radii}")
    return cx[0], cy[0], radii[0] # TODO: Smartly select the correct circle not just the first one

def main():
    settings = parse_args()
    print("Loaded settings:")
    print(settings)

    reference_size = 0
    match settings.reference_object:
        case 50:
            reference_size = 24.25
        case 1:
            reference_size = 23.25
        case 2:
            reference_size = 25.75


    img = rgb2gray(io.imread(settings.image_path))
    img_binary = (img.astype('float32') < settings.threshold).astype('float32')

    save_current_pipeline_state(img_binary, "binary_conversion")

    x, y, rad = detect_circle(img_binary)

    one_pixel_size = reference_size / (2*rad)
    log(f"One pixel is {one_pixel_size} mm")

if __name__ == "__main__":
    main()
