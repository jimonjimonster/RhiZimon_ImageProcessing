import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import threading


class ImageStitcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Minirhizotron Image Stitcher")
        self.root.geometry("600x620")

        self.source_dir = ""
        self.target_dir = ""

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Minirhizotron Image Stitcher",
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Frame for source folder
        source_frame = tk.Frame(self.root)
        source_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(source_frame, text="Source folder:", width=15, anchor='w').pack(side=tk.LEFT)
        self.source_label = tk.Label(source_frame, text="Not selected",
                                     bg="white", relief=tk.SUNKEN, anchor='w')
        self.source_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        source_btn = tk.Button(source_frame, text="Browse",
                              command=self.select_source_dir)
        source_btn.pack(side=tk.LEFT)

        # Frame for target folder
        target_frame = tk.Frame(self.root)
        target_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(target_frame, text="Target folder:", width=15, anchor='w').pack(side=tk.LEFT)
        self.target_label = tk.Label(target_frame, text="Not selected",
                                     bg="white", relief=tk.SUNKEN, anchor='w')
        self.target_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        target_btn = tk.Button(target_frame, text="Browse",
                              command=self.select_target_dir)
        target_btn.pack(side=tk.LEFT)

        # Frame for image range selection
        range_frame = tk.Frame(self.root)
        range_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(range_frame, text="Image range:", width=15, anchor='w').pack(side=tk.LEFT)

        tk.Label(range_frame, text="Start:").pack(side=tk.LEFT, padx=(10, 5))
        self.start_var = tk.StringVar(value="0")
        self.start_combo = ttk.Combobox(range_frame, textvariable=self.start_var,
                                        width=5, state='readonly')
        self.start_combo['values'] = ['0']
        self.start_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(range_frame, text="End:").pack(side=tk.LEFT, padx=(20, 5))
        self.end_var = tk.StringVar(value="39")
        self.end_combo = ttk.Combobox(range_frame, textvariable=self.end_var,
                                      width=5, state='readonly')
        self.end_combo['values'] = ['39']
        self.end_combo.pack(side=tk.LEFT, padx=5)

        # Frame for cropping - heading
        crop_title_frame = tk.Frame(self.root)
        crop_title_frame.pack(pady=(15, 5), padx=20, fill=tk.X)
        tk.Label(crop_title_frame, text="Cropping (pixels):",
                font=("Arial", 10, "bold")).pack(anchor='w')

        # Frame for top and bottom cropping
        crop_vertical_frame = tk.Frame(self.root)
        crop_vertical_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(crop_vertical_frame, text="Top:", width=15, anchor='w').pack(side=tk.LEFT)
        self.crop_top_var = tk.StringVar(value="0")
        self.crop_top_entry = tk.Entry(crop_vertical_frame, textvariable=self.crop_top_var, width=10)
        self.crop_top_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(crop_vertical_frame, text="Bottom:").pack(side=tk.LEFT, padx=(20, 5))
        self.crop_bottom_var = tk.StringVar(value="0")
        self.crop_bottom_entry = tk.Entry(crop_vertical_frame, textvariable=self.crop_bottom_var, width=10)
        self.crop_bottom_entry.pack(side=tk.LEFT, padx=5)

        # Frame for left and right cropping
        crop_horizontal_frame = tk.Frame(self.root)
        crop_horizontal_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(crop_horizontal_frame, text="Left:", width=15, anchor='w').pack(side=tk.LEFT)
        self.crop_left_var = tk.StringVar(value="0")
        self.crop_left_entry = tk.Entry(crop_horizontal_frame, textvariable=self.crop_left_var, width=10)
        self.crop_left_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(crop_horizontal_frame, text="Right:").pack(side=tk.LEFT, padx=(20, 5))
        self.crop_right_var = tk.StringVar(value="0")
        self.crop_right_entry = tk.Entry(crop_horizontal_frame, textvariable=self.crop_right_var, width=10)
        self.crop_right_entry.pack(side=tk.LEFT, padx=5)

        # Checkbox for overlap
        overlap_frame = tk.Frame(self.root)
        overlap_frame.pack(pady=10, padx=20, fill=tk.X)

        self.remove_overlap_var = tk.BooleanVar(value=False)
        self.remove_overlap_check = tk.Checkbutton(
            overlap_frame,
            text="Remove bottom edge on overlap (except last image)",
            variable=self.remove_overlap_var
        )
        self.remove_overlap_check.pack(anchor='w')

        self.dimension_label = tk.Label(self.root, text="", fg="gray")
        self.dimension_label.pack(pady=5)

        # Info Label
        self.info_label = tk.Label(self.root, text="", fg="blue")
        self.info_label.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, length=500, mode='determinate')
        self.progress.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", fg="green")
        self.status_label.pack(pady=5)

        # Start Button
        self.start_btn = tk.Button(self.root, text="Start stitching",
                                   command=self.start_stitching,
                                   bg="green", fg="white",
                                   font=("Arial", 12, "bold"),
                                   height=2, width=20)
        self.start_btn.pack(pady=20)

    def select_source_dir(self):
        directory = filedialog.askdirectory(title="Select source folder with images")
        if directory:
            self.source_dir = directory
            self.source_label.config(text=directory)
            self.update_info()

    def select_target_dir(self):
        directory = filedialog.askdirectory(title="Select target folder")
        if directory:
            self.target_dir = directory
            self.target_label.config(text=directory)

    def update_info(self):
        if self.source_dir:
            # Count image series
            series = self.find_image_series()
            if series:
                # Find min and max image numbers across all series
                all_numbers = []
                for images in series.values():
                    all_numbers.extend([img[0] for img in images])

                if all_numbers:
                    min_num = min(all_numbers)
                    max_num = max(all_numbers)

                    # Update dropdown menus
                    range_values = [str(i) for i in range(min_num, max_num + 1)]
                    self.start_combo['values'] = range_values
                    self.end_combo['values'] = range_values
                    self.start_var.set(str(min_num))
                    self.end_var.set(str(max_num))

                # Check image dimensions
                self.check_image_dimensions()

                self.info_label.config(
                    text=f"Found: {len(series)} image series with a total of {sum(len(s) for s in series.values())} images",
                    fg="blue"
                )
            else:
                self.info_label.config(text="No image series found!", fg="red")

    def find_image_series(self):
        """Finds all image series in the source folder"""
        if not self.source_dir:
            return {}

        files = [f for f in os.listdir(self.source_dir) if f.endswith('.jpg')]
        series = {}

        for filename in files:
            # Extract base name and number
            pattern = r'(.+)_(\d+)\.jpg$'
            match = re.match(pattern, filename)
            if match:
                base_name = match.group(1)
                img_num = int(match.group(2))

                if base_name not in series:
                    series[base_name] = []
                series[base_name].append((img_num, filename))

        # Sort images by number
        for base_name in series:
            series[base_name].sort(key=lambda x: x[0])

        return series

    def check_image_dimensions(self):
        """Checks whether all images have the same dimensions"""
        series = self.find_image_series()
        if not series:
            return

        dimensions = set()

        for images in series.values():
            for _, filename in images:
                filepath = os.path.join(self.source_dir, filename)
                try:
                    with Image.open(filepath) as img:
                        dimensions.add((img.width, img.height))
                except:
                    continue

        if len(dimensions) == 1:
            width, height = list(dimensions)[0]
            self.dimension_label.config(
                text=f"Image dimensions: {width} x {height} px ✓",
                fg="green"
            )
        elif len(dimensions) > 1:
            self.dimension_label.config(
                text=f"WARNING: Different dimensions found!",
                fg="red"
            )
        else:
            self.dimension_label.config(text="", fg="gray")

    def stitch_images(self, image_files, output_path, crop_values):
        """Stitches images together vertically

        crop_values: dict with keys 'top', 'bottom', 'left', 'right', 'remove_overlap'
        """
        images = []

        for i, filename in enumerate(image_files):
            filepath = os.path.join(self.source_dir, filename)
            img = Image.open(filepath)

            width, height = img.size

            # Compute crop coordinates
            left = crop_values['left']
            top = crop_values['top']
            right = width - crop_values['right']

            # On overlap: remove bottom edge only for non-last images
            if crop_values['remove_overlap'] and i < len(image_files) - 1:
                bottom = height - crop_values['bottom']
            else:
                # No overlap or last image: crop normally
                bottom = height - crop_values['bottom']

            # If no overlap removal is requested, always crop
            if not crop_values['remove_overlap']:
                bottom = height - crop_values['bottom']

            # Crop only if necessary
            if left > 0 or top > 0 or right < width or bottom < height:
                img = img.crop((left, top, right, bottom))

            images.append(img)

            # Update Progress
            progress = int((i + 1) / len(image_files) * 50)
            self.progress['value'] = progress
            self.root.update_idletasks()

        # Determine dimensions
        widths = [img.width for img in images]
        heights = [img.height for img in images]

        max_width = max(widths)
        total_height = sum(heights)

        # Check whether the image gets too large (PIL limit: 65500 pixels)
        MAX_DIMENSION = 65500
        if total_height > MAX_DIMENSION:
            raise ValueError(
                f"The stitched image would be {total_height} pixels high!\n"
                f"Maximum allowed: {MAX_DIMENSION} pixels\n\n"
                f"Solutions:\n"
                f"- Reduce the number of images (adjust start/end)\n"
                f"- Increase the overlap (crop more pixels)\n"
                f"- Split the series into several smaller groups"
            )

        if max_width > MAX_DIMENSION:
            raise ValueError(
                f"The image is {max_width} pixels wide!\n"
                f"Maximum allowed: {MAX_DIMENSION} pixels"
            )

        # Create new image
        stitched = Image.new('RGB', (max_width, total_height))

        # Paste images together
        y_offset = 0
        for i, img in enumerate(images):
            stitched.paste(img, (0, y_offset))
            y_offset += img.height

            # Update Progress
            progress = 50 + int((i + 1) / len(images) * 50)
            self.progress['value'] = progress
            self.root.update_idletasks()

        # Save result
        stitched.save(output_path, 'JPEG', quality=95)

    def start_stitching(self):
        if not self.source_dir:
            messagebox.showerror("Error", "Please select a source folder!")
            return

        if not self.target_dir:
            messagebox.showerror("Error", "Please select a target folder!")
            return

        # Run stitching in a separate thread
        thread = threading.Thread(target=self.process_stitching)
        thread.start()

    def process_stitching(self):
        try:
            self.start_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Processing...", fg="orange")

            series = self.find_image_series()

            if not series:
                messagebox.showerror("Error", "No image series found!")
                return

            # Check image dimensions
            dimensions = set()
            for images in series.values():
                for _, filename in images:
                    filepath = os.path.join(self.source_dir, filename)
                    try:
                        with Image.open(filepath) as img:
                            dimensions.add((img.width, img.height))
                    except Exception as e:
                        messagebox.showerror("Error", f"Error reading {filename}:\n{str(e)}")
                        return

            if len(dimensions) > 1:
                dim_list = "\n".join([f"{w} x {h} px" for w, h in dimensions])
                messagebox.showerror("Error",
                    f"ALL images must have the same dimensions!\n\nDimensions found:\n{dim_list}")
                return

            # Validate crop values
            try:
                crop_top = int(self.crop_top_var.get())
                crop_bottom = int(self.crop_bottom_var.get())
                crop_left = int(self.crop_left_var.get())
                crop_right = int(self.crop_right_var.get())

                if any(val < 0 for val in [crop_top, crop_bottom, crop_left, crop_right]):
                    messagebox.showerror("Error", "All crop values must be >= 0!")
                    return

                # Check that cropping is not larger than the image
                if dimensions:
                    width, height = list(dimensions)[0]
                    if crop_left + crop_right >= width:
                        messagebox.showerror("Error",
                            f"Crop left + right ({crop_left + crop_right} px) is larger than the image width ({width} px)!")
                        return
                    if crop_top + crop_bottom >= height:
                        messagebox.showerror("Error",
                            f"Crop top + bottom ({crop_top + crop_bottom} px) is larger than the image height ({height} px)!")
                        return

            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for cropping!")
                return

            crop_values = {
                'top': crop_top,
                'bottom': crop_bottom,
                'left': crop_left,
                'right': crop_right,
                'remove_overlap': self.remove_overlap_var.get()
            }

            # Get start and end values
            start_num = int(self.start_var.get())
            end_num = int(self.end_var.get())

            if start_num > end_num:
                messagebox.showerror("Error", "Start number must be less than or equal to end number!")
                return

            total_series = len(series)

            for idx, (base_name, images) in enumerate(series.items()):
                self.status_label.config(
                    text=f"Processing series {idx + 1}/{total_series}: {base_name}"
                )

                # Filter images by selected range
                filtered_images = [(num, fname) for num, fname in images
                                  if start_num <= num <= end_num]

                if not filtered_images:
                    self.status_label.config(
                        text=f"Series {base_name}: No images in range {start_num}-{end_num}"
                    )
                    continue

                # Extract file names only
                image_files = [img[1] for img in filtered_images]

                # Build output file name
                output_filename = f"{base_name}_stitched_{start_num}-{end_num}.jpg"
                output_path = os.path.join(self.target_dir, output_filename)

                # Stitch images with crop values
                self.stitch_images(image_files, output_path, crop_values)

                self.progress['value'] = int((idx + 1) / total_series * 100)

            self.status_label.config(text="Done!", fg="green")
            messagebox.showinfo("Success",
                              f"All {total_series} image series stitched successfully!")

        except Exception as e:
            self.status_label.config(text="An error occurred!", fg="red")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            self.start_btn.config(state=tk.NORMAL)
            self.progress['value'] = 0


def main():
    root = tk.Tk()
    app = ImageStitcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
