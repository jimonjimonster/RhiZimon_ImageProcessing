import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageDewarpTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Bildentzerrung mit Referenzpunkten")
        
        # Variablen
        self.reference_img = None
        self.distorted_img = None
        self.reference_points = []
        self.distorted_points = []
        self.current_mode = None  # 'reference' oder 'distorted'
        
        # Neue Variablen für Punkt-Editing und Lupe
        self.selected_point_index = None
        self.dragging = False
        self.zoom_level = 3  # Vergrößerungsfaktor für Lupe
        self.zoom_radius = 75  # Radius der Lupe in Pixel
        self.show_zoom = tk.BooleanVar(value=True)
        
        # GUI erstellen
        self.create_gui()
        
    def create_gui(self):
        # Menüleiste
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Referenzbild laden", command=self.load_reference)
        file_menu.add_command(label="Verzerrtes Bild laden", command=self.load_distorted)
        file_menu.add_separator()
        file_menu.add_command(label="Referenzpunkte laden (.npy)", command=self.load_control_points)
        file_menu.add_command(label="Referenzpunkte exportieren", command=self.export_control_points)
        file_menu.add_command(label="Entzerrtes Bild speichern", command=self.save_result)
        file_menu.add_command(label="Beenden", command=self.root.quit)
        
        # Hauptframe
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Linke Seite - Referenzbild
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="Referenzbild (nicht verzerrt)", font=('Arial', 12, 'bold')).pack()
        self.canvas_ref = tk.Canvas(left_frame, bg='gray', width=400, height=400)
        self.canvas_ref.pack(fill=tk.BOTH, expand=True)
        self.canvas_ref.bind("<Button-1>", self.on_reference_click)
        self.canvas_ref.bind("<Motion>", self.on_reference_motion)
        self.canvas_ref.bind("<Leave>", self.on_canvas_leave)
        self.canvas_ref.bind("<B1-Motion>", self.on_reference_drag)
        
        # Rechte Seite - Verzerrtes Bild
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="Verzerrtes Bild", font=('Arial', 12, 'bold')).pack()
        self.canvas_dist = tk.Canvas(right_frame, bg='gray', width=400, height=400)
        self.canvas_dist.pack(fill=tk.BOTH, expand=True)
        self.canvas_dist.bind("<Button-1>", self.on_distorted_click)
        self.canvas_dist.bind("<Motion>", self.on_distorted_motion)
        self.canvas_dist.bind("<Leave>", self.on_canvas_leave)
        self.canvas_dist.bind("<B1-Motion>", self.on_distorted_drag)
        
        # Untere Kontrollleiste
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Linke Seite - Info
        left_control = tk.Frame(control_frame)
        left_control.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.info_label = tk.Label(left_control, text="Lade zuerst beide Bilder", font=('Arial', 10))
        self.info_label.pack(anchor=tk.W)
        
        tk.Checkbutton(left_control, text="Lupe anzeigen", variable=self.show_zoom).pack(anchor=tk.W, pady=(5, 0))
        
        # Rechte Seite - Buttons in zwei Spalten
        right_control = tk.Frame(control_frame)
        right_control.pack(side=tk.RIGHT)
        
        # Linke Button-Spalte (Punkt-Verwaltung)
        left_buttons = tk.Frame(right_control)
        left_buttons.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(left_buttons, text="Letzten Punkt löschen", command=self.delete_last_point, 
                 width=20).pack(pady=2)
        tk.Button(left_buttons, text="Punkte zurücksetzen", command=self.reset_points,
                 width=20).pack(pady=2)
        
        # Rechte Button-Spalte (Entzerren)
        right_buttons = tk.Frame(right_control)
        right_buttons.pack(side=tk.LEFT)
        
        tk.Button(right_buttons, text="Entzerren (Homographie)!", command=self.dewarp_image, 
                 bg='lightgreen', font=('Arial', 10, 'bold'), width=20).pack(pady=2)
        tk.Button(right_buttons, text="Entzerren (RBF)!", command=self.dewarp_image_rbf, 
                 bg='lightblue', font=('Arial', 10, 'bold'), width=20).pack(pady=2)
        
    def load_reference(self):
        filepath = filedialog.askopenfilename(
            title="Referenzbild auswählen",
            filetypes=[("Bilddateien", "*.jpg *.jpeg *.png *.bmp"), ("Alle Dateien", "*.*")]
        )
        if filepath:
            self.reference_img = cv2.imread(filepath)
            self.reference_img = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.reference_img, self.canvas_ref)
            
            # Wenn Punkte bereits geladen wurden, zeichne sie
            if len(self.reference_points) > 0:
                self.redraw_points()
            
            self.update_info()
            
    def load_distorted(self):
        filepath = filedialog.askopenfilename(
            title="Verzerrtes Bild auswählen",
            filetypes=[("Bilddateien", "*.jpg *.jpeg *.png *.bmp"), ("Alle Dateien", "*.*")]
        )
        if filepath:
            self.distorted_img = cv2.imread(filepath)
            self.distorted_img = cv2.cvtColor(self.distorted_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.distorted_img, self.canvas_dist)
            
            # Wenn Punkte bereits geladen wurden, zeichne sie
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
            messagebox.showwarning("Warnung", "Bitte beide Bilder laden!")
            return
        
        # Prüfe ob ein existierender Punkt angeklickt wurde
        clicked_point = self.find_point_at_position(event.x, event.y, self.canvas_ref, self.reference_points)
        
        if clicked_point is not None:
            # Punkt wurde angeklickt - für Verschieben vorbereiten
            self.selected_point_index = clicked_point
            self.dragging = True
            self.current_mode = 'reference'
        else:
            # Neuen Punkt hinzufügen
            x = (event.x - self.canvas_ref.offset_x) / self.canvas_ref.scale
            y = (event.y - self.canvas_ref.offset_y) / self.canvas_ref.scale
            
            self.reference_points.append((x, y))
            self.draw_point(self.canvas_ref, event.x, event.y, len(self.reference_points))
            self.update_info()
    
    def on_distorted_click(self, event):
        if self.reference_img is None or self.distorted_img is None:
            messagebox.showwarning("Warnung", "Bitte beide Bilder laden!")
            return
        
        # Prüfe ob ein existierender Punkt angeklickt wurde
        clicked_point = self.find_point_at_position(event.x, event.y, self.canvas_dist, self.distorted_points)
        
        if clicked_point is not None:
            # Punkt wurde angeklickt - für Verschieben vorbereiten
            self.selected_point_index = clicked_point
            self.dragging = True
            self.current_mode = 'distorted'
        else:
            # Neuen Punkt hinzufügen
            x = (event.x - self.canvas_dist.offset_x) / self.canvas_dist.scale
            y = (event.y - self.canvas_dist.offset_y) / self.canvas_dist.scale
            
            self.distorted_points.append((x, y))
            self.draw_point(self.canvas_dist, event.x, event.y, len(self.distorted_points))
            self.update_info()
    
    def find_point_at_position(self, x, y, canvas, points):
        """Findet ob ein Punkt an der gegebenen Position ist"""
        threshold = 10  # Pixel-Radius zum Anklicken
        
        for i, (px, py) in enumerate(points):
            canvas_x = px * canvas.scale + canvas.offset_x
            canvas_y = py * canvas.scale + canvas.offset_y
            
            distance = np.sqrt((x - canvas_x)**2 + (y - canvas_y)**2)
            if distance < threshold:
                return i
        return None
    
    def on_reference_drag(self, event):
        if self.dragging and self.current_mode == 'reference' and self.selected_point_index is not None:
            # Aktualisiere Punktposition
            x = (event.x - self.canvas_ref.offset_x) / self.canvas_ref.scale
            y = (event.y - self.canvas_ref.offset_y) / self.canvas_ref.scale
            
            self.reference_points[self.selected_point_index] = (x, y)
            self.redraw_points()
    
    def on_distorted_drag(self, event):
        if self.dragging and self.current_mode == 'distorted' and self.selected_point_index is not None:
            # Aktualisiere Punktposition
            x = (event.x - self.canvas_dist.offset_x) / self.canvas_dist.scale
            y = (event.y - self.canvas_dist.offset_y) / self.canvas_dist.scale
            
            self.distorted_points[self.selected_point_index] = (x, y)
            self.redraw_points()
    
    def on_reference_motion(self, event):
        # Beende Drag wenn Maustaste losgelassen
        if not event.state & 0x0100:  # Left button not pressed
            self.dragging = False
            self.selected_point_index = None
        
        # Zeige Lupe
        if self.show_zoom.get() and self.reference_img is not None:
            self.show_magnifier(event, self.canvas_ref, self.reference_img, self.reference_points)
    
    def on_distorted_motion(self, event):
        # Beende Drag wenn Maustaste losgelassen
        if not event.state & 0x0100:  # Left button not pressed
            self.dragging = False
            self.selected_point_index = None
        
        # Zeige Lupe
        if self.show_zoom.get() and self.distorted_img is not None:
            self.show_magnifier(event, self.canvas_dist, self.distorted_img, self.distorted_points)
    
    def on_canvas_leave(self, event):
        """Entferne Lupe wenn Maus Canvas verlässt"""
        self.dragging = False
        self.selected_point_index = None
        # Entferne Lupe von beiden Canvas
        self.canvas_ref.delete("magnifier")
        self.canvas_dist.delete("magnifier")
    
    def show_magnifier(self, event, canvas, img, points):
        """Zeigt eine kreisförmige Lupe am Mauszeiger"""
        canvas.delete("magnifier")
        
        mouse_x = event.x
        mouse_y = event.y
        
        # Berechne Bildkoordinaten
        img_x = (mouse_x - canvas.offset_x) / canvas.scale
        img_y = (mouse_y - canvas.offset_y) / canvas.scale
        
        # Prüfe ob im Bild
        h, w = img.shape[:2]
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return
        
        # Extrahiere Bildausschnitt für Lupe
        # Nutze eine feste Pixel-Größe im Originalbild (unabhängig von der Canvas-Skalierung)
        zoom_size_pixels = 25  # Pixel im Originalbild
        x1 = max(0, int(img_x - zoom_size_pixels))
        y1 = max(0, int(img_y - zoom_size_pixels))
        x2 = min(w, int(img_x + zoom_size_pixels))
        y2 = min(h, int(img_y + zoom_size_pixels))
        
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            
            # Vergrößere auf feste Lupe-Größe (unabhängig vom Ausschnitt)
            target_size = self.zoom_radius * 2 - 20  # Etwas kleiner als Kreis
            zoomed = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Konvertiere zu PhotoImage
            zoomed_pil = Image.fromarray(zoomed)
            zoomed_tk = ImageTk.PhotoImage(zoomed_pil)
            
            # Positioniere Lupe rechts unten vom Cursor, aber Fadenkreuz direkt an Cursorspitze
            offset_x = 15
            offset_y = 15
            lupe_x = mouse_x + offset_x + self.zoom_radius
            lupe_y = mouse_y + offset_y + self.zoom_radius
            
            # Halte Lupe im Canvas
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()
            
            # Wenn rechts nicht genug Platz, zeige links
            if lupe_x + self.zoom_radius > canvas_w:
                lupe_x = mouse_x - offset_x - self.zoom_radius
            # Wenn unten nicht genug Platz, zeige oben
            if lupe_y + self.zoom_radius > canvas_h:
                lupe_y = mouse_y - offset_y - self.zoom_radius
            
            # Zeichne Lupe-Hintergrund (Kreis)
            canvas.create_oval(
                lupe_x - self.zoom_radius, lupe_y - self.zoom_radius,
                lupe_x + self.zoom_radius, lupe_y + self.zoom_radius,
                fill='white', outline='black', width=3, tags="magnifier"
            )
            
            # Zeige vergrößertes Bild
            canvas.create_image(lupe_x, lupe_y, image=zoomed_tk, tags="magnifier")
            canvas.zoomed_image = zoomed_tk  # Referenz behalten
            
            # Berechne wo in der Lupe die Mausposition ist
            # Die Maus ist im Zentrum des Ausschnitts
            center_offset_x = int((img_x - x1) / (x2 - x1) * target_size)
            center_offset_y = int((img_y - y1) / (y2 - y1) * target_size)
            
            # Position des Fadenkreuzes in der Lupe
            crosshair_x = lupe_x - target_size//2 + center_offset_x
            crosshair_y = lupe_y - target_size//2 + center_offset_y
            
            # Zeichne Fadenkreuz an der tatsächlichen Mausposition in der Lupe
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
            
            # Zeichne Linie vom Cursor zum Fadenkreuz in der Lupe
            canvas.create_line(
                mouse_x, mouse_y,
                crosshair_x, crosshair_y,
                fill='red', width=1, dash=(2, 2), tags="magnifier"
            )
        
    def draw_point(self, canvas, x, y, number):
        r = 5
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='red', outline='white', width=2, tags='point')
        # Text mit schwarzer Umrandung für bessere Sichtbarkeit
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
                self.info_label.config(text=f"⚠️ Punkte geladen ({ref_count}), aber Bilder fehlen noch")
            else:
                self.info_label.config(text="Lade zuerst beide Bilder")
        elif ref_count != dist_count:
            self.info_label.config(text=f"Punkte: Referenz={ref_count}, Verzerrt={dist_count} - Bitte gleiche Anzahl setzen!")
        elif ref_count < 4:
            self.info_label.config(text=f"Punkte gesetzt: {ref_count}/4 - Mindestens 4 Punktpaare benötigt")
        else:
            self.info_label.config(text=f"Punkte gesetzt: {ref_count} - Bereit zum Entzerren! (Export über Menü möglich)")
            
    def log_message(self, message):
        """Einfaches Logging für Export-Funktion"""
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
            messagebox.showerror("Fehler", "Mindestens 4 Punktpaare benötigt!")
            return
            
        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Fehler", "Anzahl der Punkte muss übereinstimmen!")
            return
            
        try:
            # Punkte in NumPy Arrays konvertieren
            src_pts = np.float32(self.distorted_points)
            dst_pts = np.float32(self.reference_points)
            
            # Homographie-Matrix berechnen
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Bild entzerren
            h, w = self.reference_img.shape[:2]
            self.dewarped_img = cv2.warpPerspective(self.distorted_img, H, (w, h))
            
            # Ergebnis in neuem Fenster anzeigen
            self.show_result()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Entzerrung fehlgeschlagen: {str(e)}")
    
    def dewarp_image_tps(self):
        """Thin Plate Spline Transformation - besser für Barrel Distortion"""
        if len(self.reference_points) < 4 or len(self.distorted_points) < 4:
            messagebox.showerror("Fehler", "Mindestens 4 Punktpaare benötigt!")
            return
            
        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Fehler", "Anzahl der Punkte muss übereinstimmen!")
            return
            
        try:
            src_pts = np.float32(self.distorted_points)
            dst_pts = np.float32(self.reference_points)
            
            # TPS Transformer erstellen
            tps = cv2.createThinPlateSplineShapeTransformer()
            
            # Punkte in das richtige Format bringen
            src_shape = src_pts.reshape(1, -1, 2)
            dst_shape = dst_pts.reshape(1, -1, 2)
            
            # Matches erstellen
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
            
            # TPS schätzen
            tps.estimateTransformation(dst_shape, src_shape, matches)
            
            # Bild transformieren
            h, w = self.reference_img.shape[:2]
            self.dewarped_img = np.zeros_like(self.distorted_img)
            
            # Für jeden Pixel im Zielbild die Quellposition berechnen
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 1, 2)
            
            # TPS anwenden
            transformed_coords = tps.applyTransformation(coords)[1]
            transformed_coords = transformed_coords.reshape(h, w, 2)
            
            # Remap durchführen
            map_x = transformed_coords[:, :, 0].astype(np.float32)
            map_y = transformed_coords[:, :, 1].astype(np.float32)
            
            self.dewarped_img = cv2.remap(self.distorted_img, map_x, map_y, 
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            self.show_result()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"TPS-Entzerrung fehlgeschlagen: {str(e)}\n\nVerwende stattdessen RBF-Interpolation...")
            self.dewarp_image_rbf()
            
    def dewarp_image_rbf(self):
        """Radial Basis Function Interpolation - Alternative für komplexe Verzerrungen"""
        try:
            from scipy.interpolate import Rbf
            
            src_pts = np.array(self.distorted_points)
            dst_pts = np.array(self.reference_points)
            
            # RBF für x und y Koordinaten erstellen
            rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], function='thin_plate', smooth=0)
            rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1], function='thin_plate', smooth=0)
            
            # Grid für Transformation erstellen
            h, w = self.reference_img.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            
            # Inverse Mapping: für jeden Pixel im Zielbild die Quellposition berechnen
            # Wir müssen die umgekehrte Transformation verwenden
            rbf_x_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate', smooth=0)
            rbf_y_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate', smooth=0)
            
            map_x = rbf_x_inv(x_coords, y_coords).astype(np.float32)
            map_y = rbf_y_inv(x_coords, y_coords).astype(np.float32)
            
            # Remap durchführen
            self.dewarped_img = cv2.remap(self.distorted_img, map_x, map_y, 
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            self.show_result()
            
        except ImportError:
            messagebox.showerror("Fehler", "scipy ist nicht installiert!\nInstalliere mit: pip install scipy")
        except Exception as e:
            messagebox.showerror("Fehler", f"RBF-Entzerrung fehlgeschlagen: {str(e)}")
            
    def show_result(self):
        result_window = tk.Toplevel(self.root)
        result_window.title("Entzerrtes Bild")
        result_window.geometry("800x600")
        
        canvas = tk.Canvas(result_window, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bild anzeigen
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
            messagebox.showwarning("Warnung", "Zuerst das Bild entzerren!")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Entzerrtes Bild speichern",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Alle Dateien", "*.*")]
        )
        
        if filepath:
            img_bgr = cv2.cvtColor(self.dewarped_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
            messagebox.showinfo("Erfolg", "Bild erfolgreich gespeichert!")
    
    def load_control_points(self):
        """Lädt Referenzpunkte aus .npy Datei"""
        filepath = filedialog.askopenfilename(
            title="Referenzpunkte laden",
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("Alle Dateien", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            data = np.load(filepath, allow_pickle=True).item()
            
            # Lade Punkte
            loaded_source = data['source_points']
            loaded_target = data['target_points']
            reference_shape = data.get('reference_shape', None)
            distorted_shape = data.get('distorted_shape', None)
            
            num_points = len(loaded_source)
            
            # Prüfe ob Bilder bereits geladen sind
            if self.reference_img is None or self.distorted_img is None:
                response = messagebox.askyesno(
                    "Bilder nicht geladen",
                    f"{num_points} Punktpaare gefunden.\n\n"
                    "Referenz- und verzerrtes Bild sind noch nicht geladen.\n"
                    "Möchtest du die Punkte trotzdem laden?\n\n"
                    "(Die Punkte werden gesetzt sobald du die Bilder lädst)"
                )
                if not response:
                    return
                
                # Speichere Punkte temporär
                self.reference_points = loaded_target.tolist()
                self.distorted_points = loaded_source.tolist()
                
                messagebox.showinfo(
                    "Punkte geladen",
                    f"{num_points} Punktpaare wurden geladen.\n\n"
                    "Bitte lade jetzt:\n"
                    "1. Das Referenzbild (nicht verzerrt)\n"
                    "2. Das verzerrte Bild\n\n"
                    "Die Punkte werden dann automatisch angezeigt."
                )
                self.update_info()
                return
            
            # Wenn Bilder geladen sind, prüfe Dimensionen
            current_ref_shape = self.reference_img.shape[:2]
            current_dist_shape = self.distorted_img.shape[:2]
            
            warning_msg = ""
            
            if reference_shape is not None and tuple(reference_shape) != current_ref_shape:
                warning_msg += (f"⚠️ WARNUNG: Referenzbildgröße unterscheidet sich!\n"
                               f"   Gespeichert: {reference_shape[1]}x{reference_shape[0]}\n"
                               f"   Aktuell: {current_ref_shape[1]}x{current_ref_shape[0]}\n\n")
            
            if distorted_shape is not None and tuple(distorted_shape) != current_dist_shape:
                warning_msg += (f"⚠️ WARNUNG: Verzerrte Bildgröße unterscheidet sich!\n"
                               f"   Gespeichert: {distorted_shape[1]}x{distorted_shape[0]}\n"
                               f"   Aktuell: {current_dist_shape[1]}x{current_dist_shape[0]}\n\n")
            
            if warning_msg:
                response = messagebox.askyesno(
                    "Bildgrößen unterschiedlich",
                    warning_msg + 
                    "Die Punkte passen möglicherweise nicht exakt.\n\n"
                    "Möchtest du die Punkte trotzdem laden?\n"
                    "(Du kannst sie dann manuell anpassen)"
                )
                if not response:
                    return
            
            # Lade Punkte
            self.reference_points = loaded_target.tolist()
            self.distorted_points = loaded_source.tolist()
            
            # Zeichne Punkte neu
            self.redraw_points()
            self.update_info()
            
            messagebox.showinfo(
                "Erfolgreich geladen",
                f"✓ {num_points} Punktpaare wurden geladen!\n\n"
                f"Du kannst die Punkte jetzt bearbeiten:\n"
                f"• Punkte verschieben: Anklicken und ziehen\n"
                f"• Neue Punkte: Auf freie Stelle klicken\n"
                f"• Punkte löschen: 'Letzten Punkt löschen'\n\n"
                f"Vergiss nicht, die Änderungen zu exportieren!"
            )
            
            self.log_message(f"Referenzpunkte geladen: {num_points} Punktpaare aus {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Referenzpunkte nicht laden:\n{str(e)}")
            self.log_message(f"Fehler beim Laden: {str(e)}")
    
    def export_control_points(self):
        """Exportiert die Referenzpunkte für Batch-Verarbeitung"""
        if len(self.reference_points) < 4 or len(self.distorted_points) < 4:
            messagebox.showerror("Fehler", "Mindestens 4 Punktpaare benötigt!")
            return
            
        if len(self.reference_points) != len(self.distorted_points):
            messagebox.showerror("Fehler", "Anzahl der Punkte muss übereinstimmen!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Referenzpunkte speichern",
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("Alle Dateien", "*.*")]
        )
        
        if filepath:
            # Speichere beide Punktsätze zusammen mit Bildgrößen
            data = {
                'source_points': np.array(self.distorted_points),
                'target_points': np.array(self.reference_points),
                'reference_shape': self.reference_img.shape[:2],
                'distorted_shape': self.distorted_img.shape[:2]
            }
            np.save(filepath, data)
            messagebox.showinfo("Erfolg", 
                f"Referenzpunkte gespeichert!\n\n"
                f"Anzahl Punktpaare: {len(self.reference_points)}\n"
                f"Datei: {os.path.basename(filepath)}\n\n"
                f"Diese Datei kann jetzt in imageTransform.py verwendet werden.")
            self.log_message(f"Referenzpunkte exportiert: {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")
    app = ImageDewarpTool(root)
    root.mainloop()