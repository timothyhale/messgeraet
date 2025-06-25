import argparse
import numpy as np
import cv2
import os

from skimage import io, draw, feature, transform
from skimage.util import img_as_ubyte
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

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


def compute_bounding_boxes_from_labels(labeled_image: np.ndarray):
    bounding_boxes = []
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]  # Hintergrund ignorieren

    for label in unique_labels:
        ys, xs = np.where(labeled_image == label)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()
        bounding_boxes.append((label, x, y, w, h))

    return bounding_boxes

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
    h, w = roi.shape[:2]

    # If background is bright (e.g. white), we invert
    if mean_val > 127:
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, binary = cv2.threshold(roi, 0, 255, thresh_type)
    contours, _ = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) / (h * w) > 0.9:
        # If the largest contour covers most of the area, we assume it's a background
        binary = cv2.bitwise_not(binary)
    return binary
                                     

def detect_circle(gray, binary, connected_components, min_circle_ratio=-np.inf):
    best_circle = None
    best_ratio = 0
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Full image copy for best circle

    for i, component in enumerate(connected_components):
        x, y, w, h, _ = component
        roi = gray[y:y+h, x:x+w]
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(f"./praesentation/roi_{i}.png", roi)
        cv2.imwrite(f"./praesentation/roi_{i}_binary_gaussian_c.png", binary[y:y+h, x:x+w])
        
        binary_roi = auto_otsu_threshold(roi)
        cv2.imwrite(f"./praesentation/roi_{i}_binary_otsu.png", binary_roi)
        
        contours, _ = cv2.findContours(binary_roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        approx += np.array([[x, y]])  # Shift contour back to full image coords

        (bcx, bcy), radius = cv2.minEnclosingCircle(approx)
        circle_ratio = intersection_ratio_contour_in_circle(gray.shape, approx)

        # Draw circle on ROI
        roi_center = (int(bcx - x), int(bcy - y))
        roi_radius = int(radius)
        cv2.circle(roi_color, roi_center, roi_radius, (0, 255, 0), 2)
        cv2.imwrite(f"./praesentation/roi_{i}_with_circle.png", roi_color)

        # Optional: draw also on global output_img
        cv2.circle(output_img, (int(bcx), int(bcy)), int(radius), (0, 255, 0), 20)

        # Track best circle
        if circle_ratio > min_circle_ratio and circle_ratio > best_ratio:
            best_ratio = circle_ratio
            best_circle = (bcx, bcy, radius)

    # Save full image with all circles drawn
    cv2.imwrite("./praesentation/all_circles.png", output_img)

    return best_circle

def detect_rects(binary_image, connected_components, ignore_index=-1):
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

def draw_connected_components(img, connected_components, thickness=20):
    output_img = img.copy()
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)  # Ensure output is in color

    for i, component in enumerate(connected_components):
        x, y, w, h, area = component
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
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

def match_coin_by_hist(cropped_coin_img, template_cfg):
    mask_coin = np.any(cropped_coin_img != 0, axis=2).astype(np.uint8) * 255
    hsv_coin = cv2.cvtColor(cropped_coin_img, cv2.COLOR_BGR2HSV)
    hist_coin = cv2.calcHist([hsv_coin], [0, 1], mask_coin, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_coin, hist_coin)

    template_dir = template_cfg["dir"]
    hist_map = {}
    for coin_type, coins in template_cfg.items():
        if coin_type == "dir":
            continue
        for template_file, value in coins.items():
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
            hist_map.setdefault(coin_type, {})[template_file] = hist_score
    return hist_map

def rotate_image(image, angle):
    """
    Rotates an image around its center, filling the empty space with black (0).

    Args:
        image (np.ndarray): The input image (BGR or grayscale).
        angle (float): The rotation angle in degrees. Positive values mean counter-clockwise.

    Returns:
        np.ndarray: The rotated image with black background fill.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the size of the output image (to avoid cropping)
    abs_cos = abs(rot_matrix[0, 0])
    abs_sin = abs(rot_matrix[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to consider the translation
    rot_matrix[0, 2] += bound_w / 2 - center[0]
    rot_matrix[1, 2] += bound_h / 2 - center[1]

    # Rotate and fill with black (value 0)
    rotated = cv2.warpAffine(image, rot_matrix, (bound_w, bound_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0) if image.ndim == 3 else 0)
    return rotated


def match_templates(cropped_coin_img, restricted_template):
    """
    Matches the cropped coin image against a set of templates with rotation and returns the best match.

    Args:
        cropped_coin_img (np.ndarray): The cropped coin image to match.
        restricted_template (dict): A dictionary containing template filenames and their values.

    Returns:
        tuple: The best matching template name and its value.
    """
    best_match = None
    best_score = -1

    # Convert to grayscale if needed
    if cropped_coin_img.ndim == 3:
        cropped_coin_img = cv2.cvtColor(cropped_coin_img, cv2.COLOR_BGR2GRAY)

    # Set rotation angles
    angle_step_size = 10
    angle_steps = np.arange(0, 360, angle_step_size)

    for template_name, template_value in restricted_template.items():
        template_path = os.path.join("./template", template_name)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            continue

        max_vals = []
        for angle in angle_steps:
            rotated_template = rotate_image(template_img, angle)

            # Resize the coin to match rotated template size
            resized_coin = cv2.resize(cropped_coin_img, (rotated_template.shape[1], rotated_template.shape[0]))

            result = cv2.matchTemplate(resized_coin, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, local_max_val, _, _ = cv2.minMaxLoc(result)
            max_vals.append(local_max_val)

        max_val = max(max_vals)
        print(f"{template_name}: max score across rotations = {max_val:.4f}")

        if max_val > best_score:
            best_score = max_val
            best_match = (template_name, template_value)

    return best_match

def preprocess_sobel_clahe(image):
    # Step 1: Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Step 2: Apply CLAHE (adaptive contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: Apply Sobel edge enhancement (gradient magnitude)
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    return sobel_magnitude


def preprocess_canny(img, lower=20, upper=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 1: Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 150)

    # # Step 2: Dilate edges to make them thicker and more tolerant
    # kernel = np.ones((3, 3), np.uint8)  # 2×2 is subtle, 3×3 is stronger
    # dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def match_templates_with_canny(cropped_coin_img, restricted_template):
    """
    Matches the cropped coin image against a set of templates with rotation and returns the best match.

    Args:
        cropped_coin_img (np.ndarray): The cropped coin image to match.
        restricted_template (dict): A dictionary containing template filenames and their values.

    Returns:
        tuple: The best matching template name and its value.
    """
    best_match = None
    best_score = -1

    # Convert to grayscale if needed
    if cropped_coin_img.ndim == 3:
        cropped_coin_img = cv2.cvtColor(cropped_coin_img, cv2.COLOR_BGR2GRAY)

    # Set rotation angles
    angle_step_size = 10
    angle_steps = np.arange(0, 360, angle_step_size)

    for template_name, template_value in restricted_template.items():
        template_path = os.path.join("./template", template_name)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            continue

        max_vals = []
        for angle in angle_steps:
            rotated_template = rotate_image(template_img, angle)
            template_edges = preprocess_canny(rotated_template)

            # Resize the coin to match rotated template size
            coin_edges = preprocess_canny(cropped_coin_img)
            resized_coin_edges = cv2.resize(coin_edges, (template_edges.shape[1], template_edges.shape[0]))
            result = cv2.matchTemplate(resized_coin_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            _, local_max_val, _, _ = cv2.minMaxLoc(result)
            max_vals.append(local_max_val)

        max_val = max(max_vals)
        print(f"{template_name}: max score across rotations = {max_val:.4f}")

        if max_val > best_score:
            best_score = max_val
            best_match = (template_name, template_value)

    return best_match


def match_templates_with_sobel_clahe(cropped_coin_img, restricted_template):
    """
    Matches the cropped coin image against a set of templates with rotation and returns the best match.

    Args:
        cropped_coin_img (np.ndarray): The cropped coin image to match.
        restricted_template (dict): A dictionary containing template filenames and their values.

    Returns:
        tuple: The best matching template name and its value.
    """
    best_match = None
    best_score = -1

    # Convert to grayscale if needed
    if cropped_coin_img.ndim == 3:
        cropped_coin_img = cv2.cvtColor(cropped_coin_img, cv2.COLOR_BGR2GRAY)

    # Set rotation angles
    angle_step_size = 10
    angle_steps = np.arange(0, 360, angle_step_size)

    for template_name, template_value in restricted_template.items():
        template_path = os.path.join("./template", template_name)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            continue

        max_vals = []
        for angle in angle_steps:
            rotated_template = rotate_image(template_img, angle)
            template_edges = preprocess_sobel_clahe(rotated_template)

            # Resize the coin to match rotated template size
            coin_edges = preprocess_sobel_clahe(cropped_coin_img)
            resized_coin_edges = cv2.resize(coin_edges, (template_edges.shape[1], template_edges.shape[0]))
            result = cv2.matchTemplate(resized_coin_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            _, local_max_val, _, _ = cv2.minMaxLoc(result)
            max_vals.append(local_max_val)

        max_val = max(max_vals)
        print(f"{template_name}: max score across rotations = {max_val:.4f}")

        if max_val > best_score:
            best_score = max_val
            best_match = (template_name, template_value)

    return best_match



def match_templates_orb(cropped_coin_img, template_dir, restricted_template, nfeatures=1000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(cropped_coin_img, None)

    best_match = None
    best_score = -1

    for template_name, template_value in restricted_template.items():
        template_path = os.path.join(template_dir, template_name)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            continue

        kp2, des2 = orb.detectAndCompute(template_img, None)
        if des1 is None or des2 is None:
            continue

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)

        num_matches = len(matches)
        print(f"{template_name}: {num_matches} matches")

        if num_matches > best_score:
            best_score = num_matches
            best_match = (template_name, template_value)

    return best_match

def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def match_templates_orb_clahe(cropped_coin_img, restricted_template, template_dir="./template"):
    """
    Uses ORB feature matching on CLAHE-preprocessed images to find the best template match.

    Args:
        cropped_coin_img (np.ndarray): The cropped coin image.
        restricted_template (dict): Dict with template filenames as keys and coin values as values.
        template_dir (str): Directory where template images are stored.

    Returns:
        tuple: (template filename, coin value) with the highest match score
    """
    orb = cv2.ORB_create(nfeatures=1000)
    coin_processed = preprocess_clahe(cropped_coin_img)
    kp1, des1 = orb.detectAndCompute(coin_processed, None)

    best_match = None
    best_score = -1

    for template_name, template_value in restricted_template.items():
        template_path = os.path.join(template_dir, template_name)
        template_img = cv2.imread(template_path)
        if template_img is None:
            continue

        template_processed = preprocess_clahe(template_img)
        kp2, des2 = orb.detectAndCompute(template_processed, None)

        if des1 is None or des2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        num_matches = len(matches)

        print(f"{template_name}: {num_matches} ORB matches")

        if num_matches > best_score:
            best_score = num_matches
            best_match = (template_name, template_value)

    return best_match


def draw_internal_contours(canny_img):
    # Step 1: Find contours with hierarchy
    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)  # No contours found

    # Step 2: Filter internal contours (those with a parent)
    internal_contours = [contours[i] for i, h in enumerate(hierarchy[0]) if h[3] != -1]

    # Step 3: Convert to BGR for colored drawing
    output_bgr = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)

    # Step 4: Draw internal contours in green (or blue, or random color)
    cv2.drawContours(output_bgr, internal_contours, -1, (0, 255, 0), 1)

    return output_bgr

def extract_digit_2(canny_img):
    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None

    internal_contours = [contours[i] for i, h in enumerate(hierarchy[0]) if h[3] != -1]

    best_candidate = None
    best_score = 0
    image_h, image_w = canny_img.shape

    for cnt in internal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / h if h > 0 else 0

        # Heuristic filtering
        if area < 100 or area > 3000:
            continue
        if aspect_ratio < 0.3 or aspect_ratio > 1.2:
            continue
        if y > image_h // 2:
            continue  # ignore stuff too far down

        # score = area * closeness to left side
        score = area * (1 - x / image_w)

        if score > best_score:
            best_score = score
            best_candidate = cnt

    if best_candidate is not None:
        mask = np.zeros_like(canny_img)
        cv2.drawContours(mask, [best_candidate], -1, 255, thickness=cv2.FILLED)

        x, y, w, h = cv2.boundingRect(best_candidate)
        digit_crop = mask[y:y+h, x:x+w]
        return digit_crop

    return None


def draw_rotated_rects(img, rotated_rects, color=(0, 255, 0), thickness=2):
    for rect in rotated_rects:
        if rect is None:
            continue
        box = cv2.boxPoints(rect)
        if box is not None and len(box) > 0:
            box = np.intp(box)  # safer than np.int0
            cv2.drawContours(img, [box], 0, color, thickness)
    return img


def custom_adaptive_gaussian_threshold(gray_img, block_size=15, C=5):
    """
    Implementiert adaptive Thresholding mit Gauß-Gewichtung.
    gray_img: Graustufenbild (numpy array)
    block_size: ungerade Zahl, Größe des lokalen Bereichs
    C: Konstante, die vom Mittelwert abgezogen wird
    """
    assert block_size % 2 == 1, "block_size muss ungerade sein"

    # Gauß-geglätteter Mittelwert im Nachbarschaftsfenster
    blurred = gaussian_filter(gray_img.astype(np.float32), sigma=block_size / 6.0)

    # Lokale Schwelle = geglätteter Wert - C
    thresh_img = np.where(gray_img > blurred - C, 255, 0).astype(np.uint8)
    return thresh_img


def adaptive_gaussian_threshold_with_border(gray_img, block_size=15, C=5):
    blurred = cv2.GaussianBlur(
        gray_img,
        (block_size, block_size),
        sigmaX=0,
        borderType=cv2.BORDER_REPLICATE
    )

    threshold = blurred - C
    binary = np.where(gray_img > threshold, 255, 0).astype(np.uint8)
    return binary



def main():
    settings = parse_args()

    # Preprocessing
    color_img = cv2.imread(settings.image_path)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./praesentation/gray.png", gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("./praesentation/blur.png", blur)
    thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=2
    )
    custom_gaussian_thresh = custom_adaptive_gaussian_threshold(blur, block_size=25, C=2)
    cv2.imwrite("./praesentation/gaussia_c_thresh.png", custom_gaussian_thresh)
    cv2.imwrite("./praesentation/custom_gaussian_thresh.png", thresh)
    otsu_thrseh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    cv2.imwrite("./praesentation/otsu_thresh.png", otsu_thrseh)
    canny_thresh = preprocess_canny(blur, lower=0, upper=0)
    cv2.imwrite("./praesentation/canny_thresh.png", canny_thresh)
    kernel = np.ones((7,7), np.uint8)
    thresh = iterative_median_filter(thresh, kernel_size=3, max_iterations=100)
    cv2.imwrite("./praesentation/thresh_median.png", thresh)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("./praesentation/closed.png", closed)

    # Detect Circle, Detect Components
    connected_components = filter_boxes_by_containment(closed)
    cv2.imwrite("./praesentation/ccl.png", draw_connected_components(thresh, connected_components=connected_components, thickness=20))
    best_circle = detect_circle(gray, thresh, connected_components)
    bcx, bcy, rad = best_circle

    rotated_rects = detect_rects(closed, connected_components)
    # best_circle = detect_hough_circles(thresh, min_radius=20, max_radius=100, min_circle_ratio=0.85)
    # bcx, bcy, rad, _ = best_circle
    if rad is None:
        log("No circle detected")
        return
    
    extracted_circle = extract_circular_roi(color_img, bcx, bcy, rad)
    # cv2.imshow("coin", extracted_circle)
    # cv2.imwrite("./praesentation/ccl.png", draw_rotated_rects(color_img, rotated_rects, thickness=20))
    # cv2.waitKey(0)
    template_cfg = {
        "dir": "./template",
        "copper": {
            "1_cent.png": 0.01,
            "2_cent.png": 0.02,
            "5_cent.png": 0.05,
        },
        "gold": {
            "10_cent.png": 0.1,
            "20_cent.png": 0.2,
            "50_cent.png": 0.5,
            "1_euro.png": 1.0,
            "2_euro.png": 2.0
        },
    }

    
    # # Münzenerkennung
    hist_map = match_coin_by_hist(extracted_circle, template_cfg)
    best_group = max(hist_map.items(), key=lambda item: sum(item[1].values()) / len(item[1]))[0]
    print(best_group)
    # restricted_map = template_cfg[best_group]
    # print("Hallo", restricted_map)

    # canny_coin = preprocess_canny(extracted_circle)
    # cv2.imshow("canny contours", draw_internal_contours(canny_coin))

    # cv2.imshow("canny internal contours", extract_digit_2(canny_coin))

    

    # sobel_clahe = preprocess_sobel_clahe(extracted_circle)
    # cv2.imshow("sobel clahe", sobel_clahe)
    # cv2.imshow("normal canny", preprocess_canny(extracted_circle))
    # cv2.imshow("coin", extracted_circle)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("cannyg")
    # matched_value = match_templates_with_canny(extracted_circle, template_cfg["copper"])
    # matched_value = match_templates_with_canny(extracted_circle, template_cfg["gold"])
    # print("Nothing")
    # matched_value = match_templates(extracted_circle, template_cfg["copper"])
    # matched_value = match_templates(extracted_circle, template_cfg["gold"])
    # print("Soble clahe")
    # matched_value = match_templates_with_sobel_clahe(extracted_circle, template_cfg["copper"])
    # matched_value = match_templates_with_sobel_clahe(extracted_circle, template_cfg["gold"])
    # # # print(matched_value)
    # print("ORB And clahe")
    # matched_value = match_templates_orb_clahe(extracted_circle, template_cfg["copper"])
    # matched_value = match_templates_orb_clahe(extracted_circle, template_cfg["gold"])
if __name__ == "__main__":
    main()
