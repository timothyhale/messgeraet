import cv2
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Extract coin region from image")
    parser.add_argument("--image", "-i", type=str, help="Path to input image", required=True)
    parser.add_argument("--output", "-o", type=str, help="Out Path of template", required=True)
    parser.add_argument("--thresh", "-t", type=int, help="Binarization threshold", default=230, required=False)
    parser.add_argument("--size", "-s", type=int, help="Resize size", default=512, required=False)
    args = parser.parse_args()

    image_path = os.path.expanduser(args.image)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return

    largest_cont = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_cont], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_cont)
    roi = image[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]

    roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)

    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Binary",  binary)
    cv2.imshow("Masked ROI", roi_masked)
    key_pressed = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key_pressed == ord("q"):
        roi_resized = cv2.resize(roi_masked, (args.size, args.size))
        cv2.imwrite(os.path.expanduser(args.output), roi_resized)

if __name__ == "__main__":
    main()
