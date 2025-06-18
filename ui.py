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


def draw_filtered_boxes(binary_image, min_area=500, min_intersection=0.9):
    connected_components = filter_boxes_by_containment(binary_image, min_area, min_intersection)
    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for i in range(len(connected_components)):
        x, y, w, h, _ = connected_components[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f'ID {i}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output


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
    image_path = "test_image_mid.jpg"
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

    binary_mode = False
    threshold_value = 127
    threshold_block_size = 11
    blur_size = 11
    thresh_constant = 2

    UpdateImage = False

    #     blur = cv2.GaussianBlur(img, (11, 11), 0)
    # thresh = cv2.adaptiveThreshold(
    #         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY_INV, blockSize=11, C=2
    #     )
    

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # Create a window to display

        imgui.begin("Processing Pipeline")
        # ----- Binary Conversion
        changed, CONVERT_TO_BINARY = imgui.checkbox("Convert to Binary", CONVERT_TO_BINARY)
        UpdateImage = UpdateImage or changed

        imgui.same_line()
        changed, INVERT = imgui.checkbox("Invert", INVERT)
        UpdateImage = UpdateImage or changed

        changed, blur_size = imgui.slider_int("GaussianBlur", blur_size, 3, 127)
        add_tooltip(imgui, "size of gaussion blur kernel")
        if changed:
            UpdateImage = True
            if blur_size % 2 == 0:
                blur_size = blur_size + 1

        changed, threshold_block_size = imgui.slider_int("AdaptiveThreshBlockSize", threshold_block_size, 3, 127)
        add_tooltip(imgui, "Size of a pixel neighborhood that is used to calculate a threshold value")
        if changed:
            UpdateImage = True
            if threshold_block_size % 2 == 0:
                threshold_block_size = threshold_block_size + 1

        changed, thresh_constant = imgui.slider_int("Constant", thresh_constant, 2, 127)
        add_tooltip(imgui, "Constant subtracted from the mean or weighted mean")
        UpdateImage = UpdateImage or changed

        imgui.separator()

        # ----- 
        changed, CONNECTED_COMP = imgui.checkbox("Find Connected Components", CONNECTED_COMP)
        UpdateImage = UpdateImage or changed

        imgui.separator()

        if imgui.button("Reset"):
            CONVERT_TO_BINARY = False            
            UpdateImage = True

        imgui.end()

        imgui.begin("Image Processor")

        if UpdateImage:
            img_tex.reset()
            img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_RGB2GRAY)
            if CONVERT_TO_BINARY:
                convert_to_binary(img_tex, block_size=threshold_block_size, blur_size=blur_size, c=thresh_constant, invert=INVERT)
            if CONNECTED_COMP:
                kernel = np.ones((7,7), np.uint8)
                closed = cv2.morphologyEx(img_tex.current_data, cv2.MORPH_CLOSE, kernel)
                img_tex.current_data = draw_filtered_boxes(closed)
                img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_BGR2GRAY)


            img_tex.current_data = cv2.cvtColor(img_tex.current_data, cv2.COLOR_GRAY2RGB)
            img_tex.update_texture()

            UpdateImage = False
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
