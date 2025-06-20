import argparse
import numpy as np
import cv2
import os

from skimage import io, draw, feature, transform
from skimage.util import img_as_ubyte
from dataclasses import dataclass

@dataclass
class Settings:
    threshold: int
    image_path: str
    output_step: str
    # reference_object: int

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
    # parser.add_argument(
    #     "--reference-object", "-r",
    #     type=int,
    #     choices=[2, 1, 50],
    #     required=True,
    #     help="Reference object for scale (2€, 1€ or 50 cent)"
    # )

    args = parser.parse_args()
    return Settings(
        threshold=args.threshold,
        image_path=args.image_path,
        output_step=args.output_step,
        # reference_object=args.reference_object
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

# def detect_hough_circles(gray_image, min_radius=30, max_radius=300, min_circle_ratio=0.85):
#     gray_blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)

#     # Hough-Transformation anwenden (auf Graustufenbild!)
#     circles = cv2.HoughCircles(
#         gray_blurred,
#         cv2.HOUGH_GRADIENT,
#         dp=1.2,
#         minDist=30,
#         param1=100,
#         param2=18,             # ↓ lower to catch weaker large-circle edges
#         minRadius=min_radius,
#         maxRadius=max_radius
#     )

#     filtered_circles = []

#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")

#         for (x, y, r) in circles:
#             # Maske für den erkannten Kreis erstellen
#             mask = np.zeros_like(gray_image, dtype=np.uint8)
#             cv2.circle(mask, (x, y), r, 255, -1)

#             # Kontur aus Maske extrahieren
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours:
#                 continue

#             contour = contours[0]
#             area_contour = cv2.contourArea(contour)
#             area_circle = np.pi * (r ** 2)
#             ratio = area_contour / area_circle if area_circle > 0 else 0

#             if ratio >= min_circle_ratio:
#                 filtered_circles.append((x, y, r, ratio))

#     # Optional: nach ratio sortieren
#     filtered_circles.sort(key=lambda c: (c[3], c[2]), reverse=True)

#     return filtered_circles[0]


def auto_otsu_threshold(roi):
    # Estimate if background is brighter or darker than foreground
    mean_val = np.mean(roi)

    # If background is bright (e.g. white), we invert
    if mean_val > 127:
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, binary = cv2.threshold(roi, 0, 255, thresh_type)
    return binary
                                     

def detect_circle(gray, connected_components, min_circle_ratio=-np.inf):
    best_circle = None
    best_ratio = 0

    for i, component in enumerate(connected_components):
        x, y, w, h, _ = component
        roi = gray[y:y+h, x:x+w]
        binary_roi = auto_otsu_threshold(roi)
        contours, _ = cv2.findContours(binary_roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        approx += np.array([[x, y]])

        (bcx, bcy), radius = cv2.minEnclosingCircle(approx)
        circle_ratio = intersection_ratio_contour_in_circle(gray.shape, approx)

        if circle_ratio > min_circle_ratio and circle_ratio > best_ratio:
            best_ratio = circle_ratio
            best_circle = bcx, bcy, radius
            cv2.imshow("Circle ROI", extract_circular_roi(gray, bcx, bcy, radius))

    return best_circle

def detect_rects(binary_image, connected_components, min_circle_ratio=-np.inf, ignore_index=-1):
    rects = []

    for i, component in enumerate(connected_components):
        if i == ignore_index:
            continue
        x, y, w, h, _ = component
        roi = binary_image[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue


        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        approx += np.array([[x, y]])

        rects.append(cv2.minAreaRect(approx))

    return rects



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

def coin_to_diameter(coin: int) -> float:
    match coin:
        case 50:
            reference_size = 24.25
        case 1:
            reference_size = 23.25
        case 2:
            reference_size = 25.75

    return reference_size

def extract_circular_roi(image, bcx, bcy, rad):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(bcx), int(bcy)), int(rad), 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    x, y, r = int(bcx), int(bcy), int(rad)
    cv2.imshow("mask", mask[max(0, y - r):y + r, max(0, x - r):x + r])
    return result[max(0, y - r):y + r, max(0, x - r):x + r]


def iterative_median_filter(image, kernel_size=3, max_iterations=1000):
    filtered_image = image.copy()
    for _ in range(max_iterations):
        # Apply median filter
        new_image = cv2.medianBlur(filtered_image, kernel_size)

        # Check for convergence
        if np.array_equal(new_image, filtered_image):
            break

        filtered_image = new_image
    return filtered_image

def match_coin_by_template_and_color(cropped_coin_img, template_dir, template_files):
    best_score = -1
    best_match = None

    mask_coin = np.any(cropped_coin_img != 0, axis=2).astype(np.uint8) * 255
    hsv_coin = cv2.cvtColor(cropped_coin_img, cv2.COLOR_BGR2HSV)
    hist_coin = cv2.calcHist([hsv_coin], [0, 1], mask_coin, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_coin, hist_coin)

    for template_file, value in template_files:
        template_path = os.path.join(template_dir, template_file)
        template_img = cv2.imread(template_path)
        if template_img is None:
            continue

        template_resized = cv2.resize(template_img, (cropped_coin_img.shape[1], cropped_coin_img.shape[0]))
        mask_template = np.any(template_resized != 0, axis=2).astype(np.uint8) * 255
        hsv_template = cv2.cvtColor(template_resized, cv2.COLOR_BGR2HSV)
        hist_template = cv2.calcHist([hsv_template], [0, 1], mask_template, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_template, hist_template)

        hist_score = cv2.compareHist(hist_coin, hist_template, cv2.HISTCMP_CORREL)

        print(f"Template: {template_file} | Value: {value} | hist_score: {hist_score:.4f}")

        if hist_score > best_score:
            best_score = hist_score
            best_match = (template_file, value)

    return best_match, best_score


def main():
    settings = parse_args()

    color_img = cv2.imread(settings.image_path)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )
    
    kernel = np.ones((7,7), np.uint8)
    thresh = iterative_median_filter(thresh, kernel_size=3, max_iterations=100)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    connected_components = filter_boxes_by_containment(closed)
    best_circle = detect_circle(gray, connected_components)
    bcx, bcy, rad = best_circle

    rotated_rects = detect_rects(closed, connected_components)
    # best_circle = detect_hough_circles(thresh, min_radius=20, max_radius=100, min_circle_ratio=0.85)
    # bcx, bcy, rad, _ = best_circle
    if rad is None:
        log("No circle detected")
        return
    
    extracted_circle = extract_circular_roi(color_img, bcx, bcy, rad)

    template_dir = "./template"
    template_files = [
                ("2_euro.png", 2.0), 
                ("1_euro.png", 1.0), 
                ("50_cent.png", 0.5), 
                ("20_cent.png", 0.2),
                ("10_cent.png", 0.1), 
                ("5_cent.png", 0.05), 
                ("2_cent.png", 0.02), 
                ("1_cent.png", 0.01)
                ]
    
    best_match, best_score = match_coin_by_template_and_color(extracted_circle, template_dir, template_files)
    print(f"Best match: {best_match}, Score: {best_score:.2f}")
    cv2.waitKey(0)




    # cv2.imshow("color image", cv2.resize(color_img, (0, 0), fx=0.4, fy=0.4))
    # cv2.imshow("circle", extracted_circle)
    # cv2.imshow("binary", cv2.resize(thresh, (0, 0), fx=0.2, fy=0.2))
    # cv2.imshow("closed", cv2.resize(closed, (0, 0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)
    # one_pixel_size = reference_size / (2*rad)
    # log(f"One pixel is {one_pixel_size} mm")

    # # out = draw_connected_components_with_sizes(img, connected_components, one_pixel_size)
    # # for hull in hulls:
    # #     img = draw_convex_hull_with_lengths(img, hull, one_pixel_size)

    # img = draw_rotated_rects_with_sizes(img, rotated_rects, one_pixel_size)

    # save_current_pipeline_state(img.astype("uint8"), "connected_components_with_sizes")

if __name__ == "__main__":
    main()
