import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import numpy as np
from scipy.interpolate import Rbf
import cv2
from datetime import datetime

class MinirhizotronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Minirhizotron Bildentzerrungs-Tool v2")
        self.root.geometry("900x850")
        self.root.resizable(True, True)
        
        # Variablen
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.control_points_file = tk.StringVar()
        self.processing = False
        
        # Transformation Variablen
        self.source_points = None
        self.target_points = None
        self.use_custom_points = tk.BooleanVar(value=False)
        self.output_resolution = tk.StringVar(value="original")  # "original", "reference", "custom"
        self.custom_width = tk.IntVar(value=2592)
        self.custom_height = tk.IntVar(value=1944)
        
        # Default Source und Target Points (alte Werte als Fallback)
        self.DEFAULT_SOURCE_POINTS = np.array([
            [[98.00,134.67], [721.00,195.00], [1286.00,224.00],  [1853.00,204.00],  [2464.00,152.00]],
            [[82.00,539.00], [722.00,579.00], [1285.00,596.00],  [1850.00,586.67],  [2478.00,554.67]],
            [[73.00,953.00], [722.00,957.00], [1284.00,959.00],  [1847.33,960.67],  [2482.67,959.33]],
            [[72.00,1366.00],[717.00,1336.00],[1282.00,1323.00], [1850.00,1337.33], [2483.33,1370.00]],
            [[80.00,1779.00],[709.00,1728.00],[1282.00,1704.00], [1856.00,1727.33], [2472.67,1777.33]],
        ])
        
        self.DEFAULT_TARGET_POINTS = np.array([
            [[0.0,    0.0],   [2592/4*1,      0.0], [2592/4*2,      0.0], [2592/4*3,      0.0], [2592/4*4,      0.0]],
            [[0.0, 1944/4*1], [2592/4*1, 1944/4*1], [2592/4*2, 1944/4*1], [2592/4*3, 1944/4*1], [2592/4*4, 1944/4*1]],
            [[0.0, 1944/4*2], [2592/4*1, 1944/4*2], [2592/4*2, 1944/4*2], [2592/4*3, 1944/4*2], [2592/4*4, 1944/4*2]],
            [[0.0, 1944/4*3], [2592/4*1, 1944/4*3], [2592/4*2, 1944/4*3], [2592/4*3, 1944/4*3], [2592/4*4, 1944/4*3]],
            [[0.0, 1944/4*4], [2592/4*1, 1944/4*4], [2592/4*2, 1944/4*4], [2592/4*3, 1944/4*4], [2592/4*4, 1944/4*4]],
        ])
        
        self.setup_gui()
        
    def setup_gui(self):
        # Hauptframe mit Padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguriere Grid-Gewichte für Responsivität
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Titel
        title_label = ttk.Label(main_frame, text="🔬 Minirhizotron Bildentzerrung v2", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control Points Sektion (NEU)
        control_frame = ttk.LabelFrame(main_frame, text="1️⃣ Referenzpunkte", padding="15")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        ttk.Radiobutton(control_frame, text="Standard-Referenzpunkte verwenden (25 Punktpaare)", 
                       variable=self.use_custom_points, value=False,
                       command=self.update_points_info).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        ttk.Radiobutton(control_frame, text="Eigene Referenzpunkte aus .npy Datei laden:", 
                       variable=self.use_custom_points, value=True,
                       command=self.update_points_info).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        
        points_input_frame = ttk.Frame(control_frame)
        points_input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=(20, 0), pady=(0, 8))
        points_input_frame.columnconfigure(0, weight=1)
        
        self.control_entry = ttk.Entry(points_input_frame, textvariable=self.control_points_file,
                                       font=('Helvetica', 9), state='disabled')
        self.control_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.control_btn = ttk.Button(points_input_frame, text="📂 Datei laden", 
                                      command=self.load_control_points, 
                                      state='disabled', width=15)
        self.control_btn.grid(row=0, column=1)
        
        self.control_info_label = ttk.Label(control_frame, text="✓ Standard-Punkte: 25 Punktpaare", 
                                           foreground="green", font=('Helvetica', 9))
        self.control_info_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Ausgabeauflösung Sektion (NEU)
        resolution_frame = ttk.LabelFrame(main_frame, text="2️⃣ Ausgabeauflösung", padding="15")
        resolution_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        resolution_frame.columnconfigure(0, weight=1)
        
        ttk.Radiobutton(resolution_frame, text="Wie Originalbilder (empfohlen - behält volle Auflösung)", 
                       variable=self.output_resolution, value="original").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        
        ttk.Radiobutton(resolution_frame, text="Wie Referenzbild aus .npy Datei", 
                       variable=self.output_resolution, value="reference").grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        
        custom_res_frame = ttk.Frame(resolution_frame)
        custom_res_frame.grid(row=2, column=0, sticky=tk.W, pady=(0, 0))
        
        ttk.Radiobutton(custom_res_frame, text="Benutzerdefiniert:", 
                       variable=self.output_resolution, value="custom").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(custom_res_frame, text="Breite:").grid(row=0, column=1, padx=(10, 5))
        ttk.Entry(custom_res_frame, textvariable=self.custom_width, width=10).grid(row=0, column=2, padx=(0, 15))
        ttk.Label(custom_res_frame, text="Höhe:").grid(row=0, column=3, padx=(0, 5))
        ttk.Entry(custom_res_frame, textvariable=self.custom_height, width=10).grid(row=0, column=4)
        
        # Input-Ordner Sektion
        ttk.Label(main_frame, text="3️⃣ Eingabe-Ordner:", font=('Helvetica', 10, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=(5, 5))
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_folder, 
                                     font=('Helvetica', 9))
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        input_btn = ttk.Button(input_frame, text="📁 Durchsuchen", 
                              command=self.select_input_folder, width=15)
        input_btn.grid(row=0, column=1)
        
        # Output-Ordner Sektion
        ttk.Label(main_frame, text="4️⃣ Ausgabe-Ordner:", font=('Helvetica', 10, 'bold')).grid(
            row=5, column=0, sticky=tk.W, pady=(5, 5))
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_folder,
                                      font=('Helvetica', 9))
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        output_btn = ttk.Button(output_frame, text="📁 Durchsuchen", 
                               command=self.select_output_folder, width=15)
        output_btn.grid(row=0, column=1)
        
        # Info-Label für gefundene Bilder
        self.info_label = ttk.Label(main_frame, text="", foreground="blue",
                                   font=('Helvetica', 9))
        self.info_label.grid(row=7, column=0, columnspan=3, pady=(0, 10))
        
        # Progress Bar
        ttk.Label(main_frame, text="Fortschritt:", font=('Helvetica', 10, 'bold')).grid(
            row=8, column=0, sticky=tk.W, pady=(0, 5))
        
        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=400)
        self.progress.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(main_frame, text="0%", font=('Helvetica', 9))
        self.progress_label.grid(row=10, column=0, columnspan=3, pady=(0, 15))
        
        # Start Button
        self.start_btn = ttk.Button(main_frame, text="▶ Entzerrung starten", 
                                    command=self.start_processing, 
                                    style='Accent.TButton')
        self.start_btn.grid(row=11, column=0, columnspan=3, pady=(0, 15), ipadx=20, ipady=10)
        
        # Log-Bereich
        ttk.Label(main_frame, text="Verarbeitungsprotokoll:", 
                 font=('Helvetica', 10, 'bold')).grid(row=12, column=0, sticky=tk.W, pady=(0, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=13, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(13, weight=1)
        
        self.log_text = tk.Text(log_frame, height=10, width=70, wrap=tk.WORD,
                               font=('Courier', 9), bg='#f5f5f5', fg='#333333')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Info-Box am unteren Rand
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Hinweise", padding="10")
        info_frame.grid(row=14, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        info_text = ("• Nur JPG-Dateien werden verarbeitet\n"
                    "• Entzerrte Bilder erhalten das Präfix 'corrected_'\n"
                    "• RBF-Interpolation für nicht-lineare Verzerrungen\n"
                    "• Empfohlen: 'Wie Originalbilder' für höchste Qualität")
        ttk.Label(info_frame, text=info_text, font=('Helvetica', 8), 
                 foreground='#555555').grid(row=0, column=0, sticky=tk.W)
        
        # Style konfigurieren
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Helvetica', 11, 'bold'))
        
        self.log_message("Programm gestartet. Bitte wähle Referenzpunkte, Eingabe- und Ausgabe-Ordner.")
        
    def update_points_info(self):
        """Aktualisiert UI basierend auf Auswahl"""
        if self.use_custom_points.get():
            self.control_entry.config(state='normal')
            self.control_btn.config(state='normal')
            if self.control_points_file.get():
                filename = os.path.basename(self.control_points_file.get())
                self.control_info_label.config(text=f"✓ Geladen: {filename}", foreground="green")
            else:
                self.control_info_label.config(text="⚠ Bitte .npy Datei laden", foreground="orange")
        else:
            self.control_entry.config(state='disabled')
            self.control_btn.config(state='disabled')
            self.control_info_label.config(text="✓ Standard-Punkte: 25 Punktpaare", foreground="green")
            
    def load_control_points(self):
        """Lädt Referenzpunkte aus .npy Datei"""
        filepath = filedialog.askopenfilename(
            title="Referenzpunkte-Datei auswählen",
            filetypes=[("NumPy Array", "*.npy"), ("Alle Dateien", "*.*")]
        )
        
        if filepath:
            try:
                data = np.load(filepath, allow_pickle=True).item()
                self.source_points = data['source_points']
                self.target_points = data['target_points']
                self.reference_shape = data.get('reference_shape', None)
                self.distorted_shape = data.get('distorted_shape', None)
                self.control_points_file.set(filepath)
                
                num_points = len(self.source_points)
                info_text = f"✓ Geladen: {num_points} Punktpaare"
                if self.reference_shape is not None:
                    info_text += f" | Ref: {self.reference_shape[1]}x{self.reference_shape[0]}"
                if self.distorted_shape is not None:
                    info_text += f" | Verz: {self.distorted_shape[1]}x{self.distorted_shape[0]}"
                    
                self.control_info_label.config(text=info_text, foreground="green")
                self.log_message(f"Referenzpunkte geladen: {num_points} Punktpaare")
                self.log_message(f"  Source (verzerrt): {self.source_points[:3]}...")
                self.log_message(f"  Target (Referenz): {self.target_points[:3]}...")
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Konnte Referenzpunkte nicht laden:\n{str(e)}")
                self.log_message(f"Fehler beim Laden der Referenzpunkte: {str(e)}", "ERROR")
                
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Eingabe-Ordner wählen")
        if folder:
            self.input_folder.set(folder)
            self.log_message(f"Eingabe-Ordner ausgewählt: {folder}")
            self.update_file_count()
            
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Ausgabe-Ordner wählen")
        if folder:
            self.output_folder.set(folder)
            self.log_message(f"Ausgabe-Ordner ausgewählt: {folder}")
            
    def update_file_count(self):
        if self.input_folder.get():
            jpg_files = [f for f in os.listdir(self.input_folder.get()) 
                        if f.lower().endswith('.jpg')]
            count = len(jpg_files)
            if count > 0:
                self.info_label.config(text=f"✓ {count} JPG-Bild(er) gefunden", 
                                      foreground="green")
            else:
                self.info_label.config(text="⚠ Keine JPG-Bilder gefunden", 
                                      foreground="orange")
                
    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_processing(self):
        if not self.input_folder.get() or not self.output_folder.get():
            messagebox.showwarning("Fehlende Angaben", 
                                  "Bitte wähle sowohl Eingabe- als auch Ausgabe-Ordner!")
            return
        
        if self.use_custom_points.get() and not self.control_points_file.get():
            messagebox.showwarning("Fehlende Referenzpunkte", 
                                  "Bitte lade eine .npy Datei mit Referenzpunkten!")
            return
            
        if not os.path.exists(self.input_folder.get()):
            messagebox.showerror("Fehler", "Eingabe-Ordner existiert nicht!")
            return
            
        if not os.path.exists(self.output_folder.get()):
            try:
                os.makedirs(self.output_folder.get())
                self.log_message(f"Ausgabe-Ordner erstellt: {self.output_folder.get()}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Konnte Ausgabe-Ordner nicht erstellen: {e}")
                return
        
        # Starte Verarbeitung in separatem Thread
        self.processing = True
        self.start_btn.config(state='disabled')
        
        thread = threading.Thread(target=self.process_images, daemon=True)
        thread.start()
        
    def process_images(self):
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()
        
        # Liste alle JPG-Dateien
        jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
        
        if len(jpg_files) == 0:
            self.log_message("Keine JPG-Dateien zum Verarbeiten gefunden!", "WARNING")
            self.start_btn.config(state='normal')
            self.processing = False
            return
        
        self.log_message(f"Starte Verarbeitung von {len(jpg_files)} Bild(ern)...", "INFO")
        
        # Wähle Punktsätze basierend auf Einstellung
        if self.use_custom_points.get():
            src_pts = self.source_points
            dst_pts = self.target_points
            self.log_message(f"Verwende eigene Referenzpunkte: {len(src_pts)} Punktpaare", "INFO")
            self.log_message(f"  Source-Bereich: X[{src_pts[:,0].min():.1f}-{src_pts[:,0].max():.1f}] Y[{src_pts[:,1].min():.1f}-{src_pts[:,1].max():.1f}]", "INFO")
            self.log_message(f"  Target-Bereich: X[{dst_pts[:,0].min():.1f}-{dst_pts[:,0].max():.1f}] Y[{dst_pts[:,1].min():.1f}-{dst_pts[:,1].max():.1f}]", "INFO")
        else:
            src_pts = self.DEFAULT_SOURCE_POINTS.reshape(-1, 2)
            dst_pts = self.DEFAULT_TARGET_POINTS.reshape(-1, 2)
            self.log_message("Verwende Standard-Referenzpunkte", "INFO")
        
        # Bestimme Ziel-Bildgröße basierend auf Auswahl
        first_img_path = os.path.join(input_dir, jpg_files[0])
        first_img = cv2.imread(first_img_path)
        source_h, source_w = first_img.shape[:2]
        
        resolution_mode = self.output_resolution.get()
        
        if resolution_mode == "original":
            # Verwende Original-Bildauflösung
            target_h, target_w = source_h, source_w
            self.log_message(f"Ausgabeauflösung: Original ({target_w}x{target_h})", "INFO")
            
        elif resolution_mode == "reference":
            # Verwende Referenzbildgröße aus .npy (falls vorhanden)
            if self.use_custom_points.get() and hasattr(self, 'reference_shape') and self.reference_shape is not None:
                target_h, target_w = self.reference_shape
                self.log_message(f"Ausgabeauflösung: Aus .npy Referenz ({target_w}x{target_h})", "INFO")
            else:
                target_h, target_w = source_h, source_w
                self.log_message(f"Ausgabeauflösung: Fallback auf Original ({target_w}x{target_h})", "WARNING")
                
        else:  # custom
            # Verwende benutzerdefinierte Größe
            target_w = self.custom_width.get()
            target_h = self.custom_height.get()
            self.log_message(f"Ausgabeauflösung: Benutzerdefiniert ({target_w}x{target_h})", "INFO")
        
        self.log_message(f"Quellbildgröße: {source_w}x{source_h}", "INFO")
        
        # Skaliere Referenzpunkte falls nötig
        # Die Punkte aus der .npy sind in den Original-Koordinaten der Kalibrierbilder
        # Wenn die Zielauflösung anders ist, müssen wir die target_points skalieren
        scale_x = 1.0
        scale_y = 1.0
        
        if self.use_custom_points.get() and hasattr(self, 'reference_shape') and self.reference_shape is not None:
            ref_h, ref_w = self.reference_shape
            scale_x = target_w / ref_w
            scale_y = target_h / ref_h
            
            if scale_x != 1.0 or scale_y != 1.0:
                self.log_message(f"Skaliere Target-Punkte: X={scale_x:.3f}, Y={scale_y:.3f}", "INFO")
                dst_pts_scaled = dst_pts.copy()
                dst_pts_scaled[:, 0] *= scale_x
                dst_pts_scaled[:, 1] *= scale_y
                dst_pts = dst_pts_scaled
        
        # Erstelle RBF-Interpolatoren für inverse Transformation (EINMALIG)
        # Diese Transformation mappt von Zielbild-Koordinaten zu Quellbild-Koordinaten
        try:
            self.log_message("Erstelle RBF-Interpolatoren für inverse Transformation...", "INFO")
            rbf_x_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate', smooth=0)
            rbf_y_inv = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate', smooth=0)
            self.log_message("RBF-Transformation initialisiert", "INFO")
        except Exception as e:
            self.log_message(f"Fehler bei RBF-Initialisierung: {str(e)}", "ERROR")
            messagebox.showerror("Fehler", f"Konnte Transformation nicht initialisieren:\n{str(e)}")
            self.start_btn.config(state='normal')
            self.processing = False
            return
        
        success_count = 0
        error_count = 0
        
        # Erstelle Koordinatengrid EINMALIG
        self.log_message(f"Erstelle Mapping-Grid ({target_w}x{target_h})...", "INFO")
        y_coords, x_coords = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
        
        # DEBUG: Zeige einige Testpunkte
        test_points = [(0, 0), (target_w//2, target_h//2), (target_w-1, target_h-1)]
        self.log_message("DEBUG - Testpunkte vor Transformation:", "INFO")
        for tx, ty in test_points:
            if tx < target_w and ty < target_h:
                sx = rbf_x_inv(float(tx), float(ty))
                sy = rbf_y_inv(float(tx), float(ty))
                self.log_message(f"  Target({tx},{ty}) -> Source({sx:.1f},{sy:.1f})", "INFO")
        
        map_x = rbf_x_inv(x_coords, y_coords).astype(np.float32)
        map_y = rbf_y_inv(x_coords, y_coords).astype(np.float32)
        self.log_message(f"Mapping-Grid: X-Bereich [{map_x.min():.1f}-{map_x.max():.1f}], Y-Bereich [{map_y.min():.1f}-{map_y.max():.1f}]", "INFO")
        
        for idx, filename in enumerate(jpg_files):
            try:
                self.log_message(f"Verarbeite ({idx+1}/{len(jpg_files)}): {filename}", "PROCESSING")
                
                # Lese Bild
                input_path = os.path.join(input_dir, filename)
                img = cv2.imread(input_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Wende Transformation an (verwendet das vorberechnete Mapping)
                img_corrected = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                
                # Speichere entzerrtes Bild
                output_filename = f"corrected_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                img_bgr = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, img_bgr)
                
                self.log_message(f"✓ Erfolgreich gespeichert: {output_filename}", "SUCCESS")
                success_count += 1
                
            except Exception as e:
                self.log_message(f"✗ Fehler bei {filename}: {str(e)}", "ERROR")
                error_count += 1
            
            # Update Progress Bar
            progress_value = ((idx + 1) / len(jpg_files)) * 100
            self.progress['value'] = progress_value
            self.progress_label.config(text=f"{progress_value:.0f}%")
            self.root.update_idletasks()
        
        # Abschlussmeldung
        self.log_message("="*60, "INFO")
        self.log_message(f"Verarbeitung abgeschlossen!", "INFO")
        self.log_message(f"Erfolgreich: {success_count} | Fehler: {error_count}", "INFO")
        self.log_message("="*60, "INFO")
        
        messagebox.showinfo("Fertig", 
                          f"Verarbeitung abgeschlossen!\n\n"
                          f"Erfolgreich: {success_count}\n"
                          f"Fehler: {error_count}\n\n"
                          f"Bilder gespeichert in:\n{output_dir}")
        
        self.start_btn.config(state='normal')
        self.processing = False


def main():
    root = tk.Tk()
    app = MinirhizotronGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()