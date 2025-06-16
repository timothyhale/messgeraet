import argparse
from skimage import io, draw, feature, transform

import numpy as np
import cv2
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

def filter_boxes_by_containment(binary_image, min_area=500, min_intersection=0.9):
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    kept_indices = []

    for i, stat_i in enumerate(stats[1:], start=1):  # Skip background (Label 0)
        x1, y1, w1, h1, area1 = stat_i
        if area1 < min_area:
            continue

        rect1 = (x1, y1, x1 + w1, y1 + h1)
        keep = True

        for j, stat_j in enumerate(stats[1:], start=1):
            if i == j:
                continue

            x2, y2, w2, h2, area2 = stat_j
            rect2 = (x2, y2, x2 + w2, y2 + h2)

            inter_x1 = max(rect1[0], rect2[0])
            inter_y1 = max(rect1[1], rect2[1])
            inter_x2 = min(rect1[2], rect2[2])
            inter_y2 = min(rect1[3], rect2[3])
            iw = max(0, inter_x2 - inter_x1)
            ih = max(0, inter_y2 - inter_y1)
            intersection = iw * ih

            if intersection / (w1 * h1) > min_intersection and area2 > area1:
                keep = False
                break

        if keep:
            kept_indices.append(i)

    return stats[kept_indices]


def draw_filtered_boxes(binary_image, min_area=500, min_intersection=0.9):
    connected_components = filter_boxes_by_containment(binary_image, min_area, min_intersection)
    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for i in range(len(connected_components)):
        x, y, w, h, _ = connected_components[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f'ID {i}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output


def intersection_ratio_contour_in_circle(binary_shape, contour):
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cx, cy, radius = int(cx), int(cy), int(radius)

    mask_contour = np.zeros(binary_shape, dtype=np.uint8)
    cv2.drawContours(mask_contour, [contour], -1, 255, thickness=-1)

    mask_circle = np.zeros(binary_shape, dtype=np.uint8)
    cv2.circle(mask_circle, (cx, cy), radius, 255, thickness=-1)

    intersection_mask = cv2.bitwise_and(mask_contour, mask_circle)
    intersection_pixels = cv2.countNonZero(intersection_mask)
    circle_pixels = cv2.countNonZero(mask_circle)

    if circle_pixels == 0:
        return 0.0

    ratio = intersection_pixels / circle_pixels
    return ratio
                                     

def detect_circle_with_contours(binary_image, connected_components, min_circle_ratio=-np.inf):
    best_circle = None
    hulls = []
    rotated_rects = []
    best_ratio = 0
    output_img = cv2.cvtColor((binary_image * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    for component in connected_components:
        x, y, w, h, _ = component
        roi = binary_image[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour += np.array([[x, y]])
        hull = cv2.convexHull(largest_contour)
        hulls.append(hull)

        (bcx, bcy), radius = cv2.minEnclosingCircle(largest_contour)
        rotated_rects.append(cv2.minAreaRect(largest_contour))
        circle_ratio = intersection_ratio_contour_in_circle(binary_image.shape, largest_contour)

        # cv2.drawContours(output_img, [largest_contour], -1, (0, 255, 0), thickness=-1)
        print(f"Radius: {radius:.2f}, Center: ({bcx:.1f}, {bcy:.1f}), Circle Ratio: {circle_ratio:.3f}")
        # cv2.circle(output_img, (int(bcx), int(bcy)), int(radius), (255, 0, 0), 2)
        # cv2.putText(output_img, f"{circle_ratio:.2f}", (int(bcx), int(bcy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if circle_ratio > min_circle_ratio and circle_ratio > best_ratio:
            best_ratio = circle_ratio
            best_circle = bcx, bcy, radius
        

    # cv2.imshow("Contours, Circles & Ratios", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return best_circle, hulls, rotated_rects


def draw_connected_components_with_sizes(img, connected_components, one_pixel_size_mm):
    output_img = img.copy()

    for i, component in enumerate(connected_components):
        x, y, w, h, area = component
        width_mm = w * one_pixel_size_mm
        height_mm = h * one_pixel_size_mm
        size_str = f"{width_mm:.2f} x {height_mm:.2f} mm"
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        font_scale = 0.1  # klein & dezent
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(size_str, font, font_scale, 1)

        center_x = x + w // 2
        center_y = y + h // 2

        text_org = (int(center_x - text_w / 2), int(center_y + text_h / 2))

        cv2.rectangle(output_img,
                      (text_org[0] - 2, text_org[1] - text_h - 2),
                      (text_org[0] + text_w + 2, text_org[1] + 2),
                      (0, 0, 0), -1)

        cv2.putText(output_img, size_str, text_org, font, font_scale, (255, 255, 255), 1)

    return output_img


def draw_convex_hull_with_lengths(image, hull, one_pixel_size_mm):
    output_img = image.copy()
    n = len(hull)

    for i in range(n):
        pt1 = tuple(hull[i][0])
        pt2 = tuple(hull[(i + 1) % n][0])  # Wrap around

        # Linie zeichnen (Hull)
        cv2.line(output_img, pt1, pt2, (0, 255, 0), 1)

        # Länge berechnen
        length_px = np.linalg.norm(np.array(pt1) - np.array(pt2))
        length_mm = length_px * one_pixel_size_mm

        # Mittelpunkt
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)

        # Text vorbereiten
        label = f"{length_mm:.2f}mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # sehr klein
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Text leicht oberhalb der Linie zentriert platzieren
        text_x = mid_x - text_size[0] // 2
        text_y = mid_y + text_size[1] // 2

        # Optional: schwarzer Hintergrund für Lesbarkeit
        cv2.rectangle(output_img,
                      (text_x - 1, text_y - text_size[1]),
                      (text_x + text_size[0] + 1, text_y + 2),
                      (0, 0, 0), -1)

        # Text zeichnen
        cv2.putText(output_img, label, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)

    return output_img


def draw_rotated_rects_with_sizes(img, rotated_rects, one_pixel_size_mm):
    output_img = img.copy()

    for rect in rotated_rects:
        (cx, cy), (w, h), angle = rect

        # Real-world dimensions
        width_mm = w * one_pixel_size_mm
        height_mm = h * one_pixel_size_mm
        size_str = f"{width_mm:.2f} x {height_mm:.2f} mm"

        # Get box points and draw the rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_img, [box], 0, (0, 0, 0), 2)

        # Draw size text at the center
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 4
        (text_w, text_h), _ = cv2.getTextSize(size_str, font, font_scale, thickness)

        center_x = int(cx)
        center_y = int(cy)
        text_org = (center_x - text_w // 2, center_y + text_h // 2)

        # Background box for better visibility
        cv2.rectangle(output_img,
                      (text_org[0] - 2, text_org[1] - text_h - 2),
                      (text_org[0] + text_w + 2, text_org[1] + 2),
                      (0, 0, 0), -1)

        cv2.putText(output_img, size_str, text_org, font, font_scale, (255, 255, 255), thickness)

    return output_img




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


    img = cv2.cvtColor(cv2.imread(settings.image_path), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=11, C=2
)
    
    thresh_otsu = cv2.threshold(
        img, settings.threshold, 255, cv2.THRESH_BINARY_INV
    )[1]
    
    save_current_pipeline_state(thresh, "./out/binarized gaussian c")
    save_current_pipeline_state(thresh_otsu, "./out/binarized otsu")

    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    save_current_pipeline_state(draw_filtered_boxes(closed), "./out/main_connected_components")

    connected_components = filter_boxes_by_containment(closed)
    cicrle_stats, hulls, _ = detect_circle_with_contours(thresh, connected_components)
    _, _, rotated_rects = detect_circle_with_contours(closed, connected_components)
    bcx, bcy, rad = cicrle_stats
    
    thresh_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    cv2.circle(thresh_rgb, (int(bcx), int(bcy)), int(rad), (255, 0, 0), 2)
    save_current_pipeline_state(thresh_rgb.astype("uint8"), "detected_circle_with_contours")

    if rad is None:
        log("No circle detected")
        return
    one_pixel_size = reference_size / (2*rad)
    log(f"One pixel is {one_pixel_size} mm")

    # out = draw_connected_components_with_sizes(img, connected_components, one_pixel_size)
    # for hull in hulls:
    #     img = draw_convex_hull_with_lengths(img, hull, one_pixel_size)

    img = draw_rotated_rects_with_sizes(img, rotated_rects, one_pixel_size)

    save_current_pipeline_state(img.astype("uint8"), "connected_components_with_sizes")

if __name__ == "__main__":
    main()
