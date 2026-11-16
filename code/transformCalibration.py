import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageDewarpTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dewarping with Control Points")

        # Variables
        self.reference_img = None
        self.distorted_img = None
        self.reference_points = []
        self.distorted_points = []
        self.current_mode = None  # 'reference' or 'distorted'

        # Variables for point editing and magnifier
        self.selected_point_index = None
        self.dragging = False
        self.zoom_level = 3  # Magnification factor for magnifier
        self.zoom_radius = 75  # Radius of the magnifier in pixels
        self.show_zoom = tk.BooleanVar(value=True)

        # Create GUI
        self.create_gui()

    def create_gui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load reference image", command=self.load_reference)
        file_menu.add_command(label="Load distorted image", command=self.load_distorted)
        file_menu.add_separator()
        file_menu.add_command(label="Load control points (.npy)", command=self.load_control_points)
        file_menu.add_command(label="Export control points", command=self.export_control_points)
        file_menu.add_command(label="Save dewarped image", command=self.save_result)
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - reference image
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Reference image (undistorted)", font=('Arial', 12, 'bold')).pack()
        self.canvas_ref = tk.Canvas(left_frame, bg='gray', width=400, height=400)
        self.canvas_ref.pack(fill=tk.BOTH, expand=True)
        self.canvas_ref.bind("<Button-1>", self.on_reference_click)
        self.canvas_ref.bind("<Motion>", self.on_reference_motion)
        self.canvas_ref.bind("<Leave>", self.on_canvas_leave)
        self.canvas_ref.bind("<B1-Motion>", self.on_reference_drag)

        # Right side - distorted image
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Distorted image", font=('Arial', 12, 'bold')).pack()
        self.canvas_dist = tk.Canvas(right_frame, bg='gray', width=400, height=400)
        self.canvas_dist.pack(fill=tk.BOTH, expand=True)
        self.canvas_dist.bind("<Button-1>", self.on_distorted_click)
        self.canvas_dist.bind("<Motion>", self.on_distorted_motion)
        self.canvas_dist.bind("<Leave>", self.on_canvas_leave)
        self.canvas_dist.bind("<B1-Motion>", self.on_distorted_drag)

        # Bottom control bar
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Left side - info
        left_control = tk.Frame(control_frame)
        left_control.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(left_control, text="Load both images first", font=('Arial', 10))
        self.info_label.pack(anchor=tk.W)

        tk.Checkbutton(left_control, text="Show magnifier", variable=self.show_zoom).pack(anchor=tk.W, pady=(5, 0))

        # Right side - buttons in two columns
        right_control = tk.Frame(control_frame)
        right_control.pack(side=tk.RIGHT)

        # Left button column (point management)
        left_buttons = tk.Frame(right_control)
        left_buttons.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(left_buttons, text="Delete last point", command=self.delete_last_point,
                 width=20).pack(pady=2)
        tk.Button(left_buttons, text="Reset points", command=self.reset_points,
                 width=20).pack(pady=2)

        # Right button column (dewarping)
        right_buttons = tk.Frame(right_control)
        right_buttons.pack(side=tk.LEFT)

        tk.Button(right_buttons, text="Dewarp (Homography)!", command=self.dewarp_image,
                 bg='lightgreen', font=('Arial', 10, 'bold'), width=20).pack(pady=2)
        tk.Button(right_buttons, text="Dewarp (RBF)!", command=self.dewarp_image_rbf,
                 bg='lightblue', font=('Arial', 10, 'bold'), width=20).pack(pady=2)

    def load_reference(self):
        filepath = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filepath:
            self.reference_img = cv2.imread(filepath)
            self.reference_img = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.reference_img, self.canvas_ref)

            # If points have already been loaded, draw them
            if len(self.reference_points) > 0:
                self.redraw_points()

            self.update_info()

    def load_distorted(self):
        filepath = filedialog.askopenfilename(
            title="Select distorted image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filepath:
            self.distorted_img = cv2.imread(filepath)
            self.distorted_img = cv2.cvtColor(self.distorted_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.distorted_img, self.canvas_dist)

            # If points have already been loaded, draw them
            if len(self.distorted_points) > 0:
                self.redraw_points()

            self.update_info()

    def display_image(self, img, canvas):
        h, w = img.shape[:2]
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w <= 1:
            canvas_w = 400
        if canvas_h <= 1:
            canvas_h = 400

        scale = min(canvas_w/w, canvas_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas.delete("all")
        canvas.create_image(canvas_w//2, canvas_h//2, image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk
        canvas.scale = scale
        canvas.offset_x = (canvas_w - new_w) // 2
        canvas.offset_y = (canvas_h - new_h) // 2

    def on_reference_click(self, event):
        if self.reference_img is None or self.distorted_img is None:
            messagebox.showwarning("Warning", "Please load both images!")
            return

        # Check whether an existing point was clicked
        clicked_point = self.find_point_at_position(event.x, event.y, self.canvas_ref, self.reference_points)

        if clicked_point is not None:
            # Point was clicked - prepare for moving
            self.selected_point_index = clicked_point
            self.dragging = True
            self.current_mode = 'reference'
        else:
            # Add new point
            x = (event.x - self.canvas_ref.offset_x) / self.canvas_ref.scale
            y = (event.y - self.canvas_ref.offset_y) / self.canvas_ref.scale

            self.reference_points.append((x, y))
            self.draw_point(self.canvas_ref, event.x, event.y, len(self.reference_points))
            self.update_info()

    def on_distorted_click(self, event):
        if self.reference_img is None or self.distorted_img is None:
            messagebox.showwarning("Warning", "Please load both images!")
            return

        # Check whether an existing point was clicked
        clicked_point = self.find_point_at_position(event.x, event.y, self.canvas_dist, self.distorted_points)

        if clicked_point is not None:
            # Point was clicked - prepare for moving
            self.selected_point_index = clicked_point
            self.dragging = True
            self.current_mode = 'distorted'
        else:
            # Add new point
            x = (event.x - self.canvas_dist.offset_x) / self.canvas_dist.scale
            y = (event.y - self.canvas_dist.offset_y) / self.canvas_dist.scale

            self.distorted_points.append((x, y))
            self.draw_point(self.canvas_dist, event.x, event.y, len(self.distorted_points))
            self.update_info()

    def find_point_at_position(self, x, y, canvas, points):
        """Checks whether a point is at the given position"""
        threshold = 10  # Pixel radius for clicking

        for i, (px, py) in enumerate(points):
            canvas_x = px * canvas.scale + canvas.offset_x
            canvas_y = py * canvas.scale + canvas.offset_y

            distance = np.sqrt((x - canvas_x)**2 + (y - canvas_y)**2)
            if distance < threshold:
                return i
        return None

    def on_reference_drag(self, event):
        if self.dragging and self.current_mode == 'reference' and self.selected_point_index is not None:
            # Update point position
            x = (event.x - self.canvas_ref.offset_x) / self.canvas_ref.scale
            y = (event.y - self.canvas_ref.offset_y) / self.canvas_ref.scale

            self.reference_points[self.selected_point_index] = (x, y)
            self.redraw_points()

    def on_distorted_drag(self, event):
        if self.dragging and self.current_mode == 'distorted' and self.selected_point_index is not None:
            # Update point position
            x = (event.x - self.canvas_dist.offset_x) / self.canvas_dist.scale
            y = (event.y - self.canvas_dist.offset_y) / self.canvas_dist.scale

            self.distorted_points[self.selected_point_index] = (x, y)
            self.redraw_points()

    def on_reference_motion(self, event):
        # End drag when mouse button is released
        if not event.state & 0x0100:  # Left button not pressed
            self.dragging = False
            self.selected_point_index = None

        # Show magnifier
        if self.show_zoom.get() and self.reference_img is not None:
            self.show_magnifier(event, self.canvas_ref, self.reference_img, self.reference_points)

    def on_distorted_motion(self, event):
        # End drag when mouse button is released
        if not event.state & 0x0100:  # Left button not pressed
            self.dragging = False
            self.selected_point_index = None

        # Show magnifier
        if self.show_zoom.get() and self.distorted_img is not None:
            self.show_magnifier(event, self.canvas_dist, self.distorted_img, self.distorted_points)

    def on_canvas_leave(self, event):
        """Remove magnifier when the mouse leaves the canvas"""
        self.dragging = False
        self.selected_point_index = None
        # Remove magnifier from both canvases
        self.canvas_ref.delete("magnifier")
        self.canvas_dist.delete("magnifier")

    def show_magnifier(self, event, canvas, img, points):
        """Shows a circular magnifier at the mouse cursor"""
        canvas.delete("magnifier")

        mouse_x = event.x
        mouse_y = event.y

        # Compute image coordinates
        img_x = (mouse_x - canvas.offset_x) / canvas.scale
        img_y = (mouse_y - canvas.offset_y) / canvas.scale

        # Check whether inside the image
        h, w = img.shape[:2]
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return

        # Extract image patch for magnifier
        # Use a fixed pixel size in the original image (independent of canvas scaling)
        zoom_size_pixels = 25  # Pixels in the original image
        x1 = max(0, int(img_x - zoom_size_pixels))
        y1 = max(0, int(img_y - zoom_size_pixels))
        x2 = min(w, int(img_x + zoom_size_pixels))
        y2 = min(h, int(img_y + zoom_size_pixels))

        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]

            # Enlarge to fixed magnifier size (independent of the patch)
            target_size = self.zoom_radius * 2 - 20  # Slightly smaller than the circle
            zoomed = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            # Convert to PhotoImage
            zoomed_pil = Image.fromarray(zoomed)
            zoomed_tk = ImageTk.PhotoImage(zoomed_pil)

            # Place magnifier bottom-right of the cursor, but crosshair right at the cursor tip
            offset_x = 15
            offset_y = 15
            lupe_x = mouse_x + offset_x + self.zoom_radius
            lupe_y = mouse_y + offset_y + self.zoom_radius

            # Keep magnifier inside the canvas
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            # If not enough space on the right, show on the left
            if lupe_x + self.zoom_radius > canvas_w:
                lupe_x = mouse_x - offset_x - self.zoom_radius
            # If not enough space at the bottom, show on top
            if lupe_y + self.zoom_radius > canvas_h:
                lupe_y = mouse_y - offset_y - self.zoom_radius

            # Draw magnifier background (circle)
            canvas.create_oval(
                lupe_x - self.zoom_radius, lupe_y - self.zoom_radius,
                lupe_x + self.zoom_radius, lupe_y + self.zoom_radius,
                fill='white', outline='black', width=3, tags="magnifier"
            )

            # Show magnified image
            canvas.create_image(lupe_x, lupe_y, image=zoomed_tk, tags="magnifier")
            canvas.zoomed_image = zoomed_tk  # Keep reference

            # Compute where the mouse position is inside the magnifier
            # The mouse is at the center of the patch
            center_offset_x = int((img_x - x1) / (x2 - x1) * target_size)
            center_offset_y = int((img_y - y1) / (y2 - y1) * target_size)

            # Position of the crosshair in the magnifier
            crosshair_x = lupe_x - target_size//2 + center_offset_x
            crosshair_y = lupe_y - target_size//2 + center_offset_y

            # Draw crosshair at the actual mouse position in the magnifier
            crosshair_size = 8
            canvas.create_line(
                crosshair_x - crosshair_size, crosshair_y,
                crosshair_x + crosshair_size, crosshair_y,
                fill='red', width=2, tags="magnifier"
            )
            canvas.create_line(
                crosshair_x, crosshair_y - crosshair_size,
                crosshair_x, crosshair_y + crosshair_size,
                fill='red', width=2, tags="magnifier"
            )

            # Draw line from cursor to crosshair in the magnifier
            canvas.create_line(
                mouse_x, mouse_y,
                crosshair_x, crosshair_y,
                fill='red', width=1, dash=(2, 2), tags="magnifier"
            )

    def draw_point(self, canvas, x, y, number):
        r = 5
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='red', outline='white', width=2, tags='point')
        # Text with black outline for better visibility
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
            canvas.create_text(x+dx, y-15+dy, text=str(number), fill='black',
                             font=('Arial', 12, 'bold'), tags='point')
        canvas.create_text(x, y-15, text=str(number), fill='yellow',
                         font=('Arial', 12, 'bold'), tags='point')

    def update_info(self):
        ref_count = len(self.reference_points)
        dist_count = len(self.distorted_points)

        if self.reference_img is None or self.distorted_img is None:
            if ref_count > 0:
                self.info_label.config(text=f"⚠️ Points loaded ({ref_count}), but images are still missing")
            else:
                self.info_label.config(text="Load both images first")
        elif ref_count != dist_count:
            self.info_label.config(text=f"Points: reference={ref_count}, distorted={dist_count} - Please set the same number!")
        elif ref_count < 4:
            self.info_label.config(text=f"Points set: {ref_count}/4 - At least 4 point pairs required")
        else:
            self.info_label.config(text=f"Points set: {ref_count} - Ready to dewarp! (Export available via menu)")

    def log_message(self, message):
        """Simple logging for export function"""
        print(message)

    def delete_last_point(self):
        if len(self.reference_points) > len(self.distorted_points):
            self.reference_points.pop()
            self.redraw_points()
        elif len(self.distorted_points) > 0:
            self.distorted_points.pop()
            self.redraw_points()
        self.update_info()

    def reset_points(self):
        self.reference_points = []
        self.distorted_points = []
        self.redraw_points()
        self.update_info()

    def redraw_points(self):
        if self.reference_img is not None:
            self.display_image(self.reference_img, self.canvas_ref)
            for i, (x, y) in enumerate(self.reference_points):
                canvas_x = x * self.canvas_ref.scale + self.canvas_ref.offset_x
                canvas_y = y * self.canvas_ref.scale + self.canvas_ref.offset_y
                self.draw_point(self.canvas_ref, canvas_x, canvas_y, i+1)

        if self.distorted_img is not None:
            self.display_image(self.distorted_img, self.canvas_dist)
            for i, (x, y) in enumerate(self.distorted_points):
                canvas_x = x * self.canvas_dist.scale + self.canvas_dist.offset_x
                canvas_y = y * self.canvas_dist.scale + self.canvas_dist.offset_y
                self.draw_point(self.canvas_dist, canvas_x, canvas_y, i+1)

    def dewarp_image(self):
        if len(self.reference_points) < 4 or len(self.distorted_points) < 4:
            messagebox.showerror("Error", "At least 4 point pairs required!")
            return

        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Error", "Number of points must match!")
            return

        try:
            # Convert points to NumPy arrays
            src_pts = np.float32(self.distorted_points)
            dst_pts = np.float32(self.reference_points)

            # Compute homography matrix
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Dewarp image
            h, w = self.reference_img.shape[:2]
            self.dewarped_img = cv2.warpPerspective(self.distorted_img, H, (w, h))

            # Show result in a new window
            self.show_result()

        except Exception as e:
            messagebox.showerror("Error", f"Dewarping failed: {str(e)}")

    def dewarp_image_tps(self):
        """Thin plate spline transformation - better for barrel distortion"""
        if len(self.reference_points) < 4 or len(self.distorted_points) < 4:
            messagebox.showerror("Error", "At least 4 point pairs required!")
            return

        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Error", "Number of points must match!")
            return

        try:
            src_pts = np.float32(self.distorted_points)
            dst_pts = np.float32(self.reference_points)

            # Create TPS transformer
            tps = cv2.createThinPlateSplineShapeTransformer()

            # Bring points into the correct format
            src_shape = src_pts.reshape(1, -1, 2)
            dst_shape = dst_pts.reshape(1, -1, 2)

            # Create matches
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]

            # Estimate TPS
            tps.estimateTransformation(dst_shape, src_shape, matches)

            # Transform image
            h, w = self.reference_img.shape[:2]
            self.dewarped_img = np.zeros_like(self.distorted_img)

            # Compute the source position for each pixel in the target image
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 1, 2)

            # Apply TPS
            transformed_coords = tps.applyTransformation(coords)[1]
            transformed_coords = transformed_coords.reshape(h, w, 2)

            # Perform remap
            map_x = transformed_coords[:, :, 0].astype(np.float32)
            map_y = transformed_coords[:, :, 1].astype(np.float32)

            self.dewarped_img = cv2.remap(self.distorted_img, map_x, map_y,
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            self.show_result()

        except Exception as e:
            messagebox.showerror("Error", f"TPS dewarping failed: {str(e)}\n\nUsing RBF interpolation instead...")
            self.dewarp_image_rbf()

    def dewarp_image_rbf(self):
        """Radial basis function interpolation - alternative for complex distortions"""
        try:
            from scipy.interpolate import Rbf

            src_pts = np.array(self.distorted_points)
            dst_pts = np.array(self.reference_points)

            # Create RBF for x and y coordinates
            rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], function='thin_plate', smooth=0)
            rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1], function='thin_plate', smooth=0)

            # Create grid for transformation
            h, w = self.reference_img.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

            # Inverse mapping: compute the source position for each pixel in the target image
            # We need to use the inverse transformation
            rbf_x_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate', smooth=0)
            rbf_y_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate', smooth=0)

            map_x = rbf_x_inv(x_coords, y_coords).astype(np.float32)
            map_y = rbf_y_inv(x_coords, y_coords).astype(np.float32)

            # Perform remap
            self.dewarped_img = cv2.remap(self.distorted_img, map_x, map_y,
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            self.show_result()

        except ImportError:
            messagebox.showerror("Error", "scipy is not installed!\nInstall with: pip install scipy")
        except Exception as e:
            messagebox.showerror("Error", f"RBF dewarping failed: {str(e)}")

    def show_result(self):
        result_window = tk.Toplevel(self.root)
        result_window.title("Dewarped image")
        result_window.geometry("800x600")

        canvas = tk.Canvas(result_window, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)

        # Display image
        h, w = self.dewarped_img.shape[:2]
        scale = min(800/w, 600/h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(self.dewarped_img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas.create_image(400, 300, image=img_tk)
        canvas.image = img_tk

    def save_result(self):
        if not hasattr(self, 'dewarped_img'):
            messagebox.showwarning("Warning", "Dewarp the image first!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save dewarped image",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )

        if filepath:
            img_bgr = cv2.cvtColor(self.dewarped_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
            messagebox.showinfo("Success", "Image saved successfully!")

    def load_control_points(self):
        """Loads control points from a .npy file"""
        filepath = filedialog.askopenfilename(
            title="Load control points",
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            data = np.load(filepath, allow_pickle=True).item()

            # Load points
            loaded_source = data['source_points']
            loaded_target = data['target_points']
            reference_shape = data.get('reference_shape', None)
            distorted_shape = data.get('distorted_shape', None)

            num_points = len(loaded_source)

            # Check whether images are already loaded
            if self.reference_img is None or self.distorted_img is None:
                response = messagebox.askyesno(
                    "Images not loaded",
                    f"{num_points} point pairs found.\n\n"
                    "Reference and distorted images are not loaded yet.\n"
                    "Do you want to load the points anyway?\n\n"
                    "(The points will be placed as soon as you load the images)"
                )
                if not response:
                    return

                # Store points temporarily
                self.reference_points = loaded_target.tolist()
                self.distorted_points = loaded_source.tolist()

                messagebox.showinfo(
                    "Points loaded",
                    f"{num_points} point pairs have been loaded.\n\n"
                    "Please load now:\n"
                    "1. The reference image (undistorted)\n"
                    "2. The distorted image\n\n"
                    "The points will then be displayed automatically."
                )
                self.update_info()
                return

            # If images are loaded, check dimensions
            current_ref_shape = self.reference_img.shape[:2]
            current_dist_shape = self.distorted_img.shape[:2]

            warning_msg = ""

            if reference_shape is not None and tuple(reference_shape) != current_ref_shape:
                warning_msg += (f"⚠️ WARNING: Reference image size differs!\n"
                               f"   Saved: {reference_shape[1]}x{reference_shape[0]}\n"
                               f"   Current: {current_ref_shape[1]}x{current_ref_shape[0]}\n\n")

            if distorted_shape is not None and tuple(distorted_shape) != current_dist_shape:
                warning_msg += (f"⚠️ WARNING: Distorted image size differs!\n"
                               f"   Saved: {distorted_shape[1]}x{distorted_shape[0]}\n"
                               f"   Current: {current_dist_shape[1]}x{current_dist_shape[0]}\n\n")

            if warning_msg:
                response = messagebox.askyesno(
                    "Image sizes differ",
                    warning_msg +
                    "The points may not fit exactly.\n\n"
                    "Do you want to load the points anyway?\n"
                    "(You can then adjust them manually)"
                )
                if not response:
                    return

            # Load points
            self.reference_points = loaded_target.tolist()
            self.distorted_points = loaded_source.tolist()

            # Redraw points
            self.redraw_points()
            self.update_info()

            messagebox.showinfo(
                "Loaded successfully",
                f"✓ {num_points} point pairs have been loaded!\n\n"
                f"You can now edit the points:\n"
                f"• Move points: click and drag\n"
                f"• New points: click on an empty spot\n"
                f"• Delete points: 'Delete last point'\n\n"
                f"Don't forget to export your changes!"
            )

            self.log_message(f"Control points loaded: {num_points} point pairs from {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load control points:\n{str(e)}")
            self.log_message(f"Error while loading: {str(e)}")

    def export_control_points(self):
        """Exports the control points for batch processing"""
        if len(self.reference_points) < 4 or len(self.distorted_points) < 4:
            messagebox.showerror("Error", "At least 4 point pairs required!")
            return

        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Error", "Number of points must match!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save control points",
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("All files", "*.*")]
        )

        if filepath:
            # Save both point sets together with image sizes
            data = {
                'source_points': np.array(self.distorted_points),
                'target_points': np.array(self.reference_points),
                'reference_shape': self.reference_img.shape[:2],
                'distorted_shape': self.distorted_img.shape[:2]
            }
            np.save(filepath, data)
            messagebox.showinfo("Success",
                f"Control points saved!\n\n"
                f"Number of point pairs: {len(self.reference_points)}\n"
                f"File: {os.path.basename(filepath)}\n\n"
                f"This file can now be used in imageTransform.py.")
            self.log_message(f"Control points exported: {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")
    app = ImageDewarpTool(root)
    root.mainloop()
