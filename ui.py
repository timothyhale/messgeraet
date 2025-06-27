import sys
import os

from enum import Enum
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

import cv2
import numpy as np

def impl_glfw_init(window_name="ImGui + OpenCV Image (Binary Toggle)", width=1500, height=1000):
    if not glfw.init():
        print("Could not initialize GLFW")
        sys.exit(1)

    # Request OpenGL 3.3 Core profile (adjust if needed)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # On macOS, you might need:
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(int(width), int(height), window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not create GLFW window")
        sys.exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # enable vsync
    return window

class ImageTexture:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_data = None  # RGB uint8 array, flipped vertically
        self.current_data = None
        self.width = 0
        self.height = 0
        self.channels = 0
        self.texture_id = None
        self.internal_format = None
        self.format = None
        self.load_image()

    def load_image(self):
        # Read with cv2
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to load image at path: {self.image_path}")

        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        # Convert to RGB(A)
        if channels == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.internal_format = gl.GL_RGB8
            self.format = gl.GL_RGB
        elif channels == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.internal_format = gl.GL_RGBA8
            self.format = gl.GL_RGBA
        else:
            # grayscale -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.internal_format = gl.GL_RGB8
            self.format = gl.GL_RGB
            channels = 3

        # Flip vertically for OpenGL
        #img_rgb = cv2.flip(img_rgb, 0)
        self.original_data = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        self.current_data = self.original_data.copy()
        self.width = w
        self.height = h
        self.channels = channels
        # Create texture
        self.create_texture()

    def create_texture(self):
        if self.texture_id is not None:
            gl.glDeleteTextures([self.texture_id])
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            self.internal_format,
            self.width,
            self.height,
            0,
            self.format,
            gl.GL_UNSIGNED_BYTE,
            self.current_data,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def update_texture(self):
        # Re-upload current_data
        if self.texture_id is None:
            return
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            self.internal_format,
            self.width,
            self.height,
            0,
            self.format,
            gl.GL_UNSIGNED_BYTE,
            self.current_data,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def reset(self):
        self.current_data = self.original_data.copy()
    

    def delete(self):
        if self.texture_id is not None:
            gl.glDeleteTextures([self.texture_id])
            self.texture_id = None

class State(Enum):
    Init = 0
    Binarize = 1

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


def draw_filtered_boxes(output, connected_components):
    for i, comp in enumerate(connected_components):
        x, y, w, h, _ = comp
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

def draw_rotated_rects_with_sizes(output_img, rotated_rects, one_pixel_size_mm):
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

def circle_from_component(binary_image, component):
    x, y, w, h, _ = component
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
    roi = binary_image[y:y+h, x:x+w]
    contours, _ = cv2.findContours(roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0,0,0

    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour += np.array([[x, y]])

    (bcx, bcy), radius = cv2.minEnclosingCircle(largest_contour)
    _, (width, height), _ = cv2.minAreaRect(largest_contour)
    return bcx, bcy, radius, width, height


def detect_circle_with_contours(binary_image, connected_components, min_circle_ratio=-np.inf):
    best_circle = None
    hulls = []
    rotated_rects = []
    best_ratio = 0
    output_img = cv2.cvtColor((binary_image * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    index_of_detected_circle = 0

    for idx, component in enumerate(connected_components):
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

        if circle_ratio > min_circle_ratio and circle_ratio > best_ratio:
            best_ratio = circle_ratio
            best_circle = bcx, bcy, radius
            index_of_detected_circle = idx

    return best_circle, hulls, rotated_rects, index_of_detected_circle




def convert_to_binary(image, block_size=11, blur_size=11, c=2, invert=False):
    # Convert original_data (RGB) to grayscale and binary, then to RGB
    # original_data is flipped vertically; keep that orientation
    # Convert RGB to grayscale
    # Threshold
    #_, binary = cv2.block_size(gray, block_size, 255, cv2.THRESH_BINARY)
    # Convert back to RGB

    blur = cv2.GaussianBlur(image.current_data, (blur_size, blur_size), 0)
    thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY, blockSize=block_size, C=c
        )

    image.current_data = np.ascontiguousarray(thresh, dtype=np.uint8)

def add_tooltip(imgui, text):
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text(text)
        imgui.end_tooltip()

def make_odd(num):
    res = num
    if num % 2 == 0:
        res = num + 1
    return res

def main():
    # 1. Initialize ImGui context
    imgui.create_context()

    # 2. Initialize GLFW window and renderer
    window = impl_glfw_init()

    try:
        xscale, yscale = glfw.get_window_content_scale(window)
    except AttributeError:
        # Older GLFW: fallback to framebuffer size vs window size
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        win_w, win_h = glfw.get_window_size(window)
        xscale = fb_w / win_w if win_w > 0 else 1.0
        yscale = fb_h / win_h if win_h > 0 else 1.0
    # Use xscale for uniform scaling
    io = imgui.get_io()
    io.font_global_scale = xscale
    # Optionally scale style sizes if available
    style = imgui.get_style()
    try:
        style.scale_all_sizes(xscale)
    except Exception:
        pass

    impl = GlfwRenderer(window)

    # 3. Load image via OpenCV
    image_path = sys.argv[1] 
    img_tex = None
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        print("Please update image_path to a valid file.")
    else:
        try:
            img_tex = ImageTexture(image_path)
            print(f"Loaded image '{image_path}' as texture {img_tex.texture_id}, size = {img_tex.width}x{img_tex.height}")
        except Exception as e:
            print("Error loading image:", e)
            img_tex = None

    # 4. Main loop
    CONVERT_TO_BINARY = False
    INVERT = False
    CONNECTED_COMP = False
    MORPHOLOGY = False
    ADAPTIVE = False

    binary_mode = False
    threshold_value = 127
    threshold_block_size = 11
    blur_size = 11
    thresh_constant = 2
    morph_kernel_size = 7
    morph_iterations = 1

    found_objects = []
    found_objects_ref_index = -1

    ref_object_size = 25.75
    one_pixel_size = 1

    UpdateImage = False

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # Create a window to display

        imgui.begin("Processing Pipeline")

        # ----- Moprhology
        changed, MORPHOLOGY = imgui.checkbox("Apply Morphology", MORPHOLOGY)
        UpdateImage = UpdateImage or changed

        changed, morph_kernel_size = imgui.slider_int("kernel size", morph_kernel_size, 3, 65)
        if changed:
            UpdateImage = True
            morph_kernel_size = make_odd(morph_kernel_size)

        add_tooltip(imgui, "Kernel size of morph close filter")
        UpdateImage = UpdateImage or changed

        changed, morph_iterations = imgui.slider_int("Number of iterations", morph_iterations, 1, 64)
        UpdateImage = UpdateImage or changed
        add_tooltip(imgui, "Number of times the morph filter is applied")


        imgui.separator()


        # ----- Binary Conversion
        changed, CONVERT_TO_BINARY = imgui.checkbox("Convert to Binary", CONVERT_TO_BINARY)
        UpdateImage = UpdateImage or changed

        imgui.same_line()
        changed, INVERT = imgui.checkbox("Invert", INVERT)
        UpdateImage = UpdateImage or changed

        imgui.same_line()
        changed, ADAPTIVE = imgui.checkbox("Adaptive", ADAPTIVE)
        UpdateImage = UpdateImage or changed

        if ADAPTIVE:
            changed, blur_size = imgui.slider_int("GaussianBlur", blur_size, 3, 127)
            add_tooltip(imgui, "size of gaussion blur kernel")
            if changed:
                UpdateImage = True
                blur_size = make_odd(blur_size)

            changed, threshold_block_size = imgui.slider_int("AdaptiveThreshBlockSize", threshold_block_size, 3, 127)
            add_tooltip(imgui, "Size of a pixel neighborhood that is used to calculate a threshold value")
            if changed:
                UpdateImage = True
                threshold_block_size = make_odd(threshold_block_size)

            changed, thresh_constant = imgui.slider_int("Constant", thresh_constant, -127, 127)
            add_tooltip(imgui, "Constant subtracted from the mean or weighted mean")
            UpdateImage = UpdateImage or changed

        else: 
            changed, threshold_value = imgui.slider_int("Threshold", threshold_value, 0, 255)
            add_tooltip(imgui, "Threshold size")
            UpdateImage = UpdateImage or changed

        imgui.separator()

        # ----- 
        changed, CONNECTED_COMP = imgui.checkbox("Find Connected Components", CONNECTED_COMP)
        UpdateImage = UpdateImage or changed

        imgui.separator()

        if imgui.button("Reset"):
            CONVERT_TO_BINARY = False
            INVERT = False
            CONNECTED_COMP = False
            MORPHOLOGY = False
            ADAPTIVE = False
            UpdateImage = True

        imgui.end()


        if UpdateImage:
            found_objects = []
            img_tex.reset()
            img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_RGB2GRAY)

            if MORPHOLOGY:
                kernel = np.ones((morph_kernel_size,morph_kernel_size), np.uint8)
                img_tex.current_data = cv2.morphologyEx(img_tex.current_data, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

            if CONVERT_TO_BINARY:
                if ADAPTIVE:
                    convert_to_binary(img_tex, block_size=threshold_block_size, blur_size=blur_size, c=thresh_constant, invert=INVERT)
                else:
                    operation = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
                    _, img_tex.current_data = cv2.threshold(img_tex.current_data, threshold_value, 255, operation)

            if CONNECTED_COMP:
                connected_components = filter_boxes_by_containment(img_tex.current_data)
                found_objects = connected_components

                if found_objects_ref_index == -1 or found_objects_ref_index >= len(found_objects):
                    cicrle_stats, hulls, rotated_rects, idx = detect_circle_with_contours(img_tex.current_data, connected_components)
                    if idx < len(connected_components) and idx >= 0:
                        found_objects_ref_index = idx

                    if cicrle_stats is not None:
                        bcx, bcy, rad = cicrle_stats                
                        one_pixel_size = 1
                        if rad is not None and rad != 0:
                            one_pixel_size = 25.75 / (2*rad)

                img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_GRAY2RGB)
                if found_objects_ref_index != -1:
                    x,y, radius, _, _ = circle_from_component(img_tex.current_data, connected_components[found_objects_ref_index])
                    cv2.circle(img_tex.current_data, (int(x), int(y)), int(radius), (255, 0, 0), 2)



                    #thresh_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
                    #draw_rotated_rects_with_sizes(img_tex.current_data, rotated_rects, one_pixel_size)

                    
                img_tex.current_data = draw_filtered_boxes(img_tex.current_data, connected_components)

            if not CONNECTED_COMP:
                img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_GRAY2RGB)


            img_tex.update_texture()

        UpdateImage = False

        imgui.begin("Object Detections")

        x = 0
        y = 0
        radius = 0
        for idx, obj in enumerate(found_objects):
            if imgui.button("ID{}".format(idx)):
                found_objects_ref_index = idx
                x,y, radius, _, _ = circle_from_component(img_tex.current_data, obj)
                one_pixel_size = ref_object_size / (2*radius)
                UpdateImage = True

            if idx == found_objects_ref_index:
                imgui.same_line()
                imgui.text("Ref Object Diameter:")
                imgui.same_line()
                changed, text_val = imgui.input_text("", "{}mm".format(ref_object_size), 8)
                if changed:
                    try:
                        ref_object_size = float(text_val)
                        x,y, radius, _,_ = circle_from_component(img_tex.current_data, obj)
                        #cv2.circle(img_tex.current_data, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                        one_pixel_size = ref_object_size / (2*radius)
                        UpdateImage = True
                    except ValueError:
                        pass


            else:
                imgui.same_line()
                x,y, radius, width, height = circle_from_component(img_tex.current_data, obj)
                imgui.text("{:.2f}mm X {:.2f}mm".format(width * one_pixel_size, height * one_pixel_size))

        imgui.end()


        imgui.begin("Image Processor")

        #imgui.text("Hello, world! Below is the loaded image (if any).")

        if img_tex is not None and img_tex.texture_id is not None:   
            # Display image size and image itself
            imgui.text(f"Image size: {img_tex.width} x {img_tex.height}")
            # Scale down if too large
            # e.g., fit to window width
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            max_w = fb_w - 20
            max_h = fb_h - 100
            disp_w, disp_h = img_tex.width, img_tex.height
            scale = min(1.0, min(max_w / img_tex.width, max_h / img_tex.height))
            disp_w = int(img_tex.width * scale)
            disp_h = int(img_tex.height * scale)
            imgui.image(img_tex.texture_id, disp_w, disp_h)
        else:
            imgui.text_colored("(No image loaded)", 1.0, 0.0, 0.0)

        imgui.end()

        # Rendering
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # Cleanup
    if img_tex is not None:
        img_tex.delete()
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
