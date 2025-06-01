import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import numpy as np
import  cv2
from PIL import Image, ImageTk
import pandas as pd

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijital Görüntü İşleme - Final Ödevi")
        self.root.geometry("1400x800")

        # Ana çerçeve
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Sol panel - kontroller
        self.control_panel = tk.Frame(self.main_frame, width=400, bg="#f0f0f0", padx=10, pady=10) 
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.control_panel.pack_propagate(False)

        # Sağ panel - görüntüler
        self.image_display_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.image_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Görüntü alanları
        self.original_frame = tk.LabelFrame(self.image_display_frame, text="Orijinal Görüntü", padx=5, pady=5)
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.processed_frame = tk.LabelFrame(self.image_display_frame, text="İşlenmiş Görüntü", padx=5, pady=5)
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_label = tk.Label(self.original_frame, bg="#e0e0e0") # Added bg for visibility
        self.original_label.pack(fill=tk.BOTH, expand=True)

        self.processed_label = tk.Label(self.processed_frame, bg="#e0e0e0") # Added bg for visibility
        self.processed_label.pack(fill=tk.BOTH, expand=True)

        # Veri değişkenleri
        self.original_image_data = None 
        self.original_image_pil = None 
        self.original_image_size = (0, 0) # (width, height)
        self.processed_image_data = None # Custom format
        self.processed_image_pil = None # PIL version of processed image for display
        self.current_image_cv = None # OpenCV format (height, width, 3) RGB
        self.object_features_data = [] # For Excel export

        self.create_control_panel()

    def create_control_panel(self):
        # Dosya işlemleri
        file_frame = tk.LabelFrame(self.control_panel, text="Dosya İşlemleri", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)

        tk.Button(file_frame, text="TXT Görüntü Yükle", command=self.load_txt_image, width=30).pack(pady=2, fill=tk.X)
        tk.Button(file_frame, text="Normal Görüntü Yükle", command=self.load_normal_image, width=30).pack(pady=2, fill=tk.X)
        tk.Button(file_frame, text="İşlenmiş Görüntüyü TXT Kaydet", command=self.save_txt_image, width=30).pack(pady=2, fill=tk.X)

        # Ana işlem seçimi
        operation_frame = tk.LabelFrame(self.control_panel, text="İşlem Kategorisi", padx=5, pady=5)
        operation_frame.pack(fill=tk.X, pady=5)

        self.main_operation_var = tk.StringVar(value="geometric")
        operations = [
            ("Geometrik Dönüşümler", "geometric"),
            ("S-Curve Kontrast Güçlendirme", "scurve"),
            ("Hough Transform", "hough"),
            ("Deblurring", "deblur"),
            ("Nesne Sayma ve Özellik Çıkarma", "object_detection")
        ]

        for text, value in operations:
            tk.Radiobutton(operation_frame, text=text, variable=self.main_operation_var, 
                           value=value, command=self.update_operation_panels, tristatevalue="x").pack(anchor=tk.W) # tristatevalue to allow deselection if needed by logic

        # Alt işlem panelleri
        self.create_geometric_panel()
        self.create_scurve_panel()
        self.create_hough_panel()
        self.create_deblur_panel()
        self.create_object_detection_panel()

        # Uygula butonu
        tk.Button(self.control_panel, text="İşlemi Uygula", command=self.apply_operation, 
                  width=30, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=10, fill=tk.X)

        self.update_operation_panels()

    def create_geometric_panel(self):
        """Geometrik dönüşümler paneli"""
        self.geometric_frame = tk.LabelFrame(self.control_panel, text="Geometrik Dönüşümler", padx=5, pady=5)
        
        self.geometric_operation_var = tk.StringVar(value="resize_larger")
        operations = [
            ("Büyütme", "resize_larger"),
            ("Küçültme", "resize_smaller"),
            ("Zoom In", "zoom_in"),
            ("Zoom Out", "zoom_out"),
            ("Döndürme", "rotate")
        ]

        for text, value in operations:
            tk.Radiobutton(self.geometric_frame, text=text, variable=self.geometric_operation_var, 
                           value=value, command=self.update_geometric_params, tristatevalue="x").pack(anchor=tk.W)

        # İnterpolasyon yöntemi
        interp_frame = tk.Frame(self.geometric_frame)
        interp_frame.pack(fill=tk.X, pady=5)
        tk.Label(interp_frame, text="İnterpolasyon:").pack(anchor=tk.W)
        
        self.interp_var = tk.StringVar(value="nearest")
        interp_methods = [("En Yakın Komşu", "nearest"), ("Bilinear", "bilinear"), 
                          ("Bicubic", "bicubic"), ("Ortalama", "average")]
        
        for text, value in interp_methods:
            tk.Radiobutton(interp_frame, text=text, variable=self.interp_var, value=value, tristatevalue="x").pack(anchor=tk.W)

        # Parametreler
        self.geometric_params_frame = tk.Frame(self.geometric_frame)
        self.geometric_params_frame.pack(fill=tk.X, pady=5)
        
        self.geometric_params = {}
        self.create_geometric_params_widgets() # Renamed for clarity

    def create_geometric_params_widgets(self):
        """Geometrik parametre widget'larını oluştur"""
        params_config = [ # Changed to params_config for clarity
            ("scale", "Ölçek Faktörü:", "2.0"),
            ("zoom_x", "Merkez X (0-1):", "0.5"),
            ("zoom_y", "Merkez Y (0-1):", "0.5"),
            ("zoom_scale", "Zoom Ölçeği:", "2.0"),
            ("angle", "Açı (Derece):", "45")
        ]

        for param_name, label_text, default_value in params_config:
            frame = tk.Frame(self.geometric_params_frame)
            tk.Label(frame, text=label_text, width=18, anchor=tk.W).pack(side=tk.LEFT) # Increased width
            entry = tk.Entry(frame, width=12) # Increased width
            entry.insert(0, default_value)
            entry.pack(side=tk.LEFT, padx=2)
            self.geometric_params[param_name] = {"frame": frame, "entry": entry}
    
    def create_scurve_panel(self):
        """S-Curve kontrast güçlendirme paneli"""
        self.scurve_frame = tk.LabelFrame(self.control_panel, text="S-Curve Kontrast Güçlendirme", padx=5, pady=5)
        
        self.scurve_type_var = tk.StringVar(value="standard")
        scurve_types = [
            ("Standart Sigmoid", "standard"),
            ("Yatay Kaydırılmış Sigmoid", "shifted"),
            ("Eğimli Sigmoid", "slope_based"), # Renamed to avoid conflict with parameter name
            ("Özel Fonksiyon (tanh)", "custom_tanh") # Clarified function
        ]

        for text, value in scurve_types:
            tk.Radiobutton(self.scurve_frame, text=text, variable=self.scurve_type_var, value=value, tristatevalue="x").pack(anchor=tk.W)

        params_frame = tk.Frame(self.scurve_frame)
        params_frame.pack(fill=tk.X, pady=5)

        tk.Label(params_frame, text="Eğim (a):", width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.slope_entry = tk.Entry(params_frame, width=10)
        self.slope_entry.insert(0, "5.0")
        self.slope_entry.pack(side=tk.LEFT, padx=2)
        tk.Frame(params_frame).pack(side=tk.LEFT, fill=tk.X, expand=True) # Spacer

        tk.Label(params_frame, text="Kayma (b):", width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.shift_entry = tk.Entry(params_frame, width=10)
        self.shift_entry.insert(0, "0.3") # Center for standard sigmoid is 0.5, so shift is relative to that or absolute depending on formula
        self.shift_entry.pack(side=tk.LEFT, padx=2)


    def create_hough_panel(self):
        """Hough Transform paneli"""
        self.hough_frame = tk.LabelFrame(self.control_panel, text="Hough Transform", padx=5, pady=5)
        
        self.hough_type_var = tk.StringVar(value="lines")
        hough_types = [
            ("Çizgi Tespiti", "lines"),
            ("Göz Tespiti", "eyes")
        ]

        for text, value in hough_types:
            tk.Radiobutton(self.hough_frame, text=text, variable=self.hough_type_var, value=value, tristatevalue="x").pack(anchor=tk.W)

        params_frame = tk.Frame(self.hough_frame)
        params_frame.pack(fill=tk.X, pady=5)

        row1_frame = tk.Frame(params_frame)
        row1_frame.pack(fill=tk.X)
        tk.Label(row1_frame, text="Threshold:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.hough_threshold_entry = tk.Entry(row1_frame, width=10)
        self.hough_threshold_entry.insert(0, "100")
        self.hough_threshold_entry.pack(side=tk.LEFT, padx=2)

        row2_frame = tk.Frame(params_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        tk.Label(row2_frame, text="Min Line Length:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.min_line_length_entry = tk.Entry(row2_frame, width=10)
        self.min_line_length_entry.insert(0, "50")
        self.min_line_length_entry.pack(side=tk.LEFT, padx=2)

        row3_frame = tk.Frame(params_frame)
        row3_frame.pack(fill=tk.X, pady=2)
        tk.Label(row3_frame, text="Max Line Gap:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.max_line_gap_entry = tk.Entry(row3_frame, width=10) # Added Max Line Gap
        self.max_line_gap_entry.insert(0, "10")
        self.max_line_gap_entry.pack(side=tk.LEFT, padx=2)


    def create_deblur_panel(self):
        """Deblurring paneli"""
        self.deblur_frame = tk.LabelFrame(self.control_panel, text="Deblurring (Motion Blur)", padx=5, pady=5)
        
        
        params_frame = tk.Frame(self.deblur_frame)
        params_frame.pack(fill=tk.X, pady=5)

        row1_frame = tk.Frame(params_frame)
        row1_frame.pack(fill=tk.X)
        tk.Label(row1_frame, text="Kernel Boyutu:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.kernel_size_entry = tk.Entry(row1_frame, width=10)
        self.kernel_size_entry.insert(0, "15")
        self.kernel_size_entry.pack(side=tk.LEFT, padx=2)

        row2_frame = tk.Frame(params_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        tk.Label(row2_frame, text="Açı (Derece):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.blur_angle_entry = tk.Entry(row2_frame, width=10)
        self.blur_angle_entry.insert(0, "0") # Example: 45 for diagonal blur
        self.blur_angle_entry.pack(side=tk.LEFT, padx=2)

    def create_object_detection_panel(self):
        """Nesne tespit paneli"""
        self.object_frame = tk.LabelFrame(self.control_panel, text="Nesne Tespiti (Koyu Yeşil)", padx=5, pady=5)
        
        
        params_frame = tk.Frame(self.object_frame)
        params_frame.pack(fill=tk.X, pady=5)

        row1_frame = tk.Frame(params_frame)
        row1_frame.pack(fill=tk.X)
        tk.Label(row1_frame, text="HSV Alt Sınır (H,S,V):", width=20, anchor=tk.W).pack(side=tk.LEFT)
        self.hsv_lower_entry = tk.Entry(params_frame, width=15)
        self.hsv_lower_entry.insert(0, "30,40,40") # Adjusted typical green range
        self.hsv_lower_entry.pack(anchor=tk.W, pady=2)

        row2_frame = tk.Frame(params_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        tk.Label(row2_frame, text="HSV Üst Sınır (H,S,V):", width=20, anchor=tk.W).pack(side=tk.LEFT)
        self.hsv_upper_entry = tk.Entry(params_frame, width=15)
        self.hsv_upper_entry.insert(0, "90,255,255") # Adjusted typical green range
        self.hsv_upper_entry.pack(anchor=tk.W, pady=2)

        tk.Button(params_frame, text="Özellikleri Excel'e Kaydet", command=self.save_to_excel, width=25).pack(pady=10, fill=tk.X)


    def update_operation_panels(self):
        """İşlem panellerini güncelle"""
        for frame in [self.geometric_frame, self.scurve_frame, self.hough_frame, 
                        self.deblur_frame, self.object_frame]:
            if frame.winfo_exists(): # Check if widget exists before pack_forget
                 frame.pack_forget()

        operation = self.main_operation_var.get()
        if operation == "geometric":
            self.geometric_frame.pack(fill=tk.X, pady=5)
            self.update_geometric_params()
        elif operation == "scurve":
            self.scurve_frame.pack(fill=tk.X, pady=5)
        elif operation == "hough":
            self.hough_frame.pack(fill=tk.X, pady=5)
        elif operation == "deblur":
            self.deblur_frame.pack(fill=tk.X, pady=5)
        elif operation == "object_detection":
            self.object_frame.pack(fill=tk.X, pady=5)

    def update_geometric_params(self):
        """Geometrik parametreleri güncelle"""
        operation = self.geometric_operation_var.get()
        
        for param_name, param_info in self.geometric_params.items():
            if param_info["frame"].winfo_exists():
                param_info["frame"].pack_forget()

        if operation in ["resize_larger", "resize_smaller"]:
            self.geometric_params["scale"]["frame"].pack(fill=tk.X, pady=2)
        elif operation in ["zoom_in", "zoom_out"]:
            for param in ["zoom_x", "zoom_y", "zoom_scale"]:
                self.geometric_params[param]["frame"].pack(fill=tk.X, pady=2)
        elif operation == "rotate":
            self.geometric_params["angle"]["frame"].pack(fill=tk.X, pady=2)

    def display_image(self, image_data_txt, label_widget, is_original=False):
        """
        Displays an image from custom TXT format [width][height][RGB] onto a Tkinter Label.
        Updates self.original_image_pil or self.processed_image_pil.
        """
        if not image_data_txt:
            label_widget.config(image=None)
            label_widget.image = None # Keep a reference
            if is_original:
                self.original_image_pil = None
            else:
                self.processed_image_pil = None
            return

        try:
            width = len(image_data_txt)
            if width == 0:
                raise ValueError("Image data is empty (width is 0).")
            height = len(image_data_txt[0])
            if height == 0:
                raise ValueError("Image data is empty (height is 0).")

            # Create a PIL Image
            pil_image = Image.new("RGB", (width, height))
            pixels = pil_image.load()

            for x in range(width):
                for y in range(height):
                    r, g, b = image_data_txt[x][y]
                    pixels[x, y] = (int(r), int(g), int(b))
            
            if is_original:
                self.original_image_pil = pil_image
            else:
                self.processed_image_pil = pil_image

            # Resize for display if too large (optional, good for UI)
            max_display_width = label_widget.winfo_width() if label_widget.winfo_width() > 10 else 500 # Fallback
            max_display_height = label_widget.winfo_height() if label_widget.winfo_height() > 10 else 500 # Fallback
            
            img_copy = pil_image.copy() # Work on a copy for resizing
            img_copy.thumbnail((max_display_width, max_display_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_copy)
            label_widget.config(image=img_tk)
            label_widget.image = img_tk # Keep a reference to avoid garbage collection
        except Exception as e:
            messagebox.showerror("Görüntüleme Hatası", f"Görüntü gösterilirken hata: {e}")
            label_widget.config(image=None)
            label_widget.image = None
            if is_original:
                self.original_image_pil = None
            else:
                self.processed_image_pil = None


    def load_normal_image(self):
        """Normal görüntü dosyasını yükle"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.current_image_cv = cv2.imread(filepath)
                if self.current_image_cv is None:
                    raise ValueError("OpenCV görüntüyü okuyamadı. Dosya formatı desteklenmiyor veya dosya bozuk olabilir.")
                
                self.current_image_cv = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
                
                height_cv, width_cv, _ = self.current_image_cv.shape
                self.original_image_size = (width_cv, height_cv)
                
                # Convert to custom TXT format
                self.original_image_data = [[[0, 0, 0] for _ in range(height_cv)] for _ in range(width_cv)]
                for y_cv in range(height_cv):
                    for x_cv in range(width_cv):
                        r, g, b = self.current_image_cv[y_cv, x_cv]
                        self.original_image_data[x_cv][y_cv] = [int(r), int(g), int(b)]
                
                self.display_image(self.original_image_data, self.original_label, is_original=True)
                self.processed_image_data = None # Clear processed image
                self.display_image(None, self.processed_label) # Clear processed image display
            except Exception as e:
                messagebox.showerror("Hata", f"Normal görüntü yüklenirken hata: {e}")
                self.current_image_cv = None
                self.original_image_data = None
                self.display_image(None, self.original_label, is_original=True)


    def load_txt_image(self):
        """TXT formatındaki görüntü dosyasını yükler."""
        filepath = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    raise ValueError("TXT dosyası boş.")

                size_str = lines[0].strip().split()
                if len(size_str) < 2:
                    raise ValueError("TXT dosyasının ilk satırında boyut bilgisi eksik.")
                width, height = int(size_str[0]), int(size_str[1])
                self.original_image_size = (width, height)
                
                self.original_image_data = [[[0, 0, 0] for _ in range(height)] for _ in range(width)]
                self.current_image_cv = np.zeros((height, width, 3), dtype=np.uint8) # OpenCV format (H, W, C)
                
                data_lines = lines[1:]
                if len(data_lines) < width * height:
                    messagebox.showwarning("Uyarı", f"TXT dosyasındaki piksel sayısı ({len(data_lines)}) beklenen ({width*height}) sayıdan az. Görüntü eksik olabilir.")

                pixel_index = 0
                for y_coord in range(height): # Iterate y first for row-major order in file
                    for x_coord in range(width): # Iterate x second
                        if pixel_index < len(data_lines):
                            color_str = data_lines[pixel_index].strip().split()
                            if len(color_str) < 3:
                                # Fill with black if data is malformed for this pixel
                                r, g, b = 0,0,0
                                print(f"Warning: Malformed pixel data at index {pixel_index} in TXT file. Using black.")
                            else:
                                r, g, b = int(color_str[0]), int(color_str[1]), int(color_str[2])
                            
                            # Custom format: data[x][y]
                            self.original_image_data[x_coord][y_coord] = [r, g, b]
                            # OpenCV format: cv_image[y][x]
                            self.current_image_cv[y_coord, x_coord] = [r, g, b]
                            pixel_index += 1
                        else: # Not enough data lines, fill rest with black
                            self.original_image_data[x_coord][y_coord] = [0,0,0]
                            self.current_image_cv[y_coord, x_coord] = [0,0,0]

                self.display_image(self.original_image_data, self.original_label, is_original=True)
                self.processed_image_data = None
                self.display_image(None, self.processed_label)
            except Exception as e:
                messagebox.showerror("Hata", f"TXT dosyası yüklenirken hata: {e}")
                self.original_image_data = None
                self.current_image_cv = None
                self.display_image(None, self.original_label, is_original=True)

    def save_txt_image(self):
        """İşlenmiş görüntüyü TXT formatında kaydeder."""
        if self.processed_image_data is None:
            messagebox.showerror("Hata", "Kaydedilecek işlenmiş görüntü yok.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                width = len(self.processed_image_data)
                height = len(self.processed_image_data[0])
                with open(filepath, 'w') as f:
                    f.write(f"{width} {height}\n")
                    # Pixels are written row by row (y first, then x)
                    for y in range(height):
                        for x in range(width):
                            r, g, b = self.processed_image_data[x][y]
                            f.write(f"{r} {g} {b}\n")
                messagebox.showinfo("Başarılı", "Görüntü TXT olarak kaydedildi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya kaydedilirken hata: {e}")
    
    def convert_cv_to_txt(self, cv_image):
        """OpenCV görüntüsünü (H, W, C - RGB) özel TXT formatına (W, H, RGB) dönüştürür ve görüntüler."""
        if cv_image is None:
            self.processed_image_data = None
            self.display_image(None, self.processed_label)
            return

        try:
            height_cv, width_cv = cv_image.shape[:2]
            
            if len(cv_image.shape) == 2: # Grayscale
                temp_txt_data = [[[0,0,0] for _ in range(height_cv)] for _ in range(width_cv)]
                for y_cv in range(height_cv):
                    for x_cv in range(width_cv):
                        val = int(cv_image[y_cv, x_cv])
                        temp_txt_data[x_cv][y_cv] = [val, val, val]
            elif cv_image.shape[2] == 3: # RGB
                temp_txt_data = [[[0,0,0] for _ in range(height_cv)] for _ in range(width_cv)]
                for y_cv in range(height_cv):
                    for x_cv in range(width_cv):
                        r, g, b = cv_image[y_cv, x_cv]
                        temp_txt_data[x_cv][y_cv] = [int(r), int(g), int(b)]
            else: # e.g. RGBA
                messagebox.showwarning("Uyarı", "Beklenmedik kanal sayısına sahip görüntü, RGB'ye dönüştürülüyor.")
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB) # Example for RGBA
                height_cv, width_cv = rgb_image.shape[:2]
                temp_txt_data = [[[0,0,0] for _ in range(height_cv)] for _ in range(width_cv)]
                for y_cv in range(height_cv):
                    for x_cv in range(width_cv):
                        r, g, b = rgb_image[y_cv, x_cv]
                        temp_txt_data[x_cv][y_cv] = [int(r), int(g), int(b)]

            self.processed_image_data = temp_txt_data
            self.display_image(self.processed_image_data, self.processed_label)
        except Exception as e:
            messagebox.showerror("Dönüştürme Hatası", f"OpenCV'den TXT'ye dönüştürülürken hata: {e}")
            self.processed_image_data = None
            self.display_image(None, self.processed_label)


    def get_pixel_value(self, image_data, x, y, width, height):
        """Gets pixel value from custom TXT image_data, with boundary clamping."""
        x_clamped = max(0, min(int(round(x)), width - 1))
        y_clamped = max(0, min(int(round(y)), height - 1))
        return image_data[x_clamped][y_clamped]

    def interpolate_nearest(self, image_data, x_new, y_new, orig_w, orig_h):
        return self.get_pixel_value(image_data, x_new, y_new, orig_w, orig_h)

    def interpolate_bilinear(self, image_data, x_new, y_new, orig_w, orig_h):
        x1, y1 = int(x_new), int(y_new)
        x2, y2 = min(x1 + 1, orig_w - 1), min(y1 + 1, orig_h - 1)

        dx, dy = x_new - x1, y_new - y1

        p1 = self.get_pixel_value(image_data, x1, y1, orig_w, orig_h)
        p2 = self.get_pixel_value(image_data, x2, y1, orig_w, orig_h)
        p3 = self.get_pixel_value(image_data, x1, y2, orig_w, orig_h)
        p4 = self.get_pixel_value(image_data, x2, y2, orig_w, orig_h)

        interpolated_pixel = [0, 0, 0]
        for i in range(3): # R, G, B
            val = (1 - dx) * (1 - dy) * p1[i] + \
                  dx * (1 - dy) * p2[i] + \
                  (1 - dx) * dy * p3[i] + \
                  dx * dy * p4[i]
            interpolated_pixel[i] = int(round(val))
        return interpolated_pixel

    def interpolate_bicubic_1d(self, p0, p1, p2, p3, x):
        """1D Bicubic interpolation kernel (Catmull-Rom variant for simplicity here)."""
        a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
        b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
        c = -0.5 * p0 + 0.5 * p2
        d = p1
        val = a * x**3 + b * x**2 + c * x + d
        return max(0, min(255, int(round(val))))


    def interpolate_bicubic(self, image_data, x_new, y_new, orig_w, orig_h):
        x_int, y_int = int(x_new), int(y_new)
        dx, dy = x_new - x_int, y_new - y_int

        interpolated_pixel = [0, 0, 0]
        for c in range(3): # R, G, B
            rows_interp = []
            for j_offset in range(-1, 3): # 4 rows for bicubic
                y_sample = y_int + j_offset
                p = [self.get_pixel_value(image_data, x_int + i_offset, y_sample, orig_w, orig_h)[c] for i_offset in range(-1, 3)] # 4 points
                rows_interp.append(self.interpolate_bicubic_1d(p[0], p[1], p[2], p[3], dx))
            
            final_val = self.interpolate_bicubic_1d(rows_interp[0], rows_interp[1], rows_interp[2], rows_interp[3], dy)
            interpolated_pixel[c] = final_val
        return interpolated_pixel
        
    def interpolate_average(self, image_data, x_new, y_new, orig_w, orig_h):
        x_floor, y_floor = int(x_new), int(y_new)
        pixels_to_average = []
        for dx_offset in [0, 1]:
            for dy_offset in [0, 1]:
                pixels_to_average.append(self.get_pixel_value(image_data, x_floor + dx_offset, y_floor + dy_offset, orig_w, orig_h))
        
        avg_pixel = [0,0,0]
        for i in range(3): # R,G,B
            avg_pixel[i] = int(round(sum(p[i] for p in pixels_to_average) / len(pixels_to_average)))
        return avg_pixel

    def resize_image(self, image_data, scale_factor, interpolation_method):
        if not image_data: return None
        orig_w = len(image_data)
        orig_h = len(image_data[0])

        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))

        if new_w <= 0 or new_h <= 0:
            messagebox.showerror("Hata", "Yeni boyutlar geçersiz (0 veya negatif).")
            return None

        resized_data = [[[0,0,0] for _ in range(new_h)] for _ in range(new_w)]
        
        interp_func = getattr(self, f"interpolate_{interpolation_method}", self.interpolate_nearest)

        for new_x in range(new_w):
            for new_y in range(new_h):
                orig_x = new_x / scale_factor
                orig_y = new_y / scale_factor
                resized_data[new_x][new_y] = interp_func(image_data, orig_x, orig_y, orig_w, orig_h)
        
        self.original_image_size = (new_w, new_h) # Update size if it's the original being transformed
        return resized_data

    def zoom(self, image_data, factor, center_x_px, center_y_px, interpolation_method, zoom_in=True):
        if not image_data: return None
        orig_w = len(image_data)
        orig_h = len(image_data[0])
        new_w, new_h = orig_w, orig_h # Output size is same as original for zoom
        zoomed_data = [[[0,0,0] for _ in range(new_h)] for _ in range(new_w)]
        interp_func = getattr(self, f"interpolate_{interpolation_method}", self.interpolate_nearest)

        if zoom_in:
            src_w = orig_w / factor
            src_h = orig_h / factor
            src_x_start = center_x_px - src_w / 2
            src_y_start = center_y_px - src_h / 2

            for new_x_disp in range(new_w): # Coordinates in the output display
                for new_y_disp in range(new_h):
                    orig_x_src = src_x_start + (new_x_disp / new_w) * src_w
                    orig_y_src = src_y_start + (new_y_disp / new_h) * src_h
                    
                    if 0 <= orig_x_src < orig_w and 0 <= orig_y_src < orig_h:
                         zoomed_data[new_x_disp][new_y_disp] = interp_func(image_data, orig_x_src, orig_y_src, orig_w, orig_h)
                    else: 
                        zoomed_data[new_x_disp][new_y_disp] = [0,0,0] # Black
        else: 
            scale_down_factor = 1.0 / factor
            
            for new_x_disp in range(new_w):
                for new_y_disp in range(new_h):
                    orig_x = (new_x_disp - new_w/2) / scale_down_factor + center_x_px
                    orig_y = (new_y_disp - new_h/2) / scale_down_factor + center_y_px

                    if 0 <= orig_x < orig_w and 0 <= orig_y < orig_h:
                        zoomed_data[new_x_disp][new_y_disp] = interp_func(image_data, orig_x, orig_y, orig_w, orig_h)
                    else: # Padded area
                        zoomed_data[new_x_disp][new_y_disp] = [128,128,128] # Gray padding


        return zoomed_data

    def rotate_image(self, image_data, angle_degrees, interpolation_method):
        if not image_data: return None
        orig_w = len(image_data)
        orig_h = len(image_data[0])
        
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

       
        new_w, new_h = orig_w, orig_h 
        rotated_data = [[[0,0,0] for _ in range(new_h)] for _ in range(new_w)]
        
        center_x_orig, center_y_orig = orig_w / 2, orig_h / 2
        center_x_new, center_y_new = new_w / 2, new_h / 2

        interp_func = getattr(self, f"interpolate_{interpolation_method}", self.interpolate_nearest)

        for new_x in range(new_w):
            for new_y in range(new_h):
               
                x_temp = new_x - center_x_new
                y_temp = new_y - center_y_new

                orig_x = x_temp * cos_a + y_temp * sin_a + center_x_orig
                orig_y = -x_temp * sin_a + y_temp * cos_a + center_y_orig

                if 0 <= orig_x < orig_w and 0 <= orig_y < orig_h:
                    rotated_data[new_x][new_y] = interp_func(image_data, orig_x, orig_y, orig_w, orig_h)
                else:
                    rotated_data[new_x][new_y] = [0,0,0] # Black for outside pixels
        
        return rotated_data


    # --- Main Operation Application ---
    def apply_operation(self):
        """Seçilen işlemi uygula"""
        is_geo_op = self.main_operation_var.get() == "geometric"
        
        if not is_geo_op and self.current_image_cv is None:
             messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin (OpenCV işlemleri için).")
             return
        if is_geo_op and self.original_image_data is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin (Geometrik işlemler için).")
            return

        operation = self.main_operation_var.get()
        
        try:
            if operation == "geometric":
                self.apply_geometric_operation()
            elif operation == "scurve":
                self.apply_scurve_operation()
            elif operation == "hough":
                self.apply_hough_operation()
            elif operation == "deblur":
                self.apply_deblur_operation()
            elif operation == "object_detection":
                self.apply_object_detection()
        except Exception as e:
            messagebox.showerror("Hata", f"İşlem uygulanırken hata: {e}")
            import traceback
            print(f"Error during {operation}: {e}\n{traceback.format_exc()}")


    def apply_geometric_operation(self):
        """Geometrik dönüşüm uygula (custom TXT formatı üzerinde çalışır)"""
        if self.original_image_data is None:
            messagebox.showerror("Hata", "Geometrik işlem için orijinal görüntü yüklenmemiş.")
            return

        geo_operation = self.geometric_operation_var.get()
        interp_method = self.interp_var.get()
        
        temp_processed_data = None # Store result here

        if geo_operation in ["resize_larger", "resize_smaller"]:
            scale = float(self.geometric_params["scale"]["entry"].get())
            if scale <= 0:
                messagebox.showerror("Hata", "Ölçek faktörü pozitif olmalıdır.")
                return
            if geo_operation == "resize_smaller":
                scale = 1.0 / scale
            temp_processed_data = self.resize_image(self.original_image_data, scale, interp_method)
            
        elif geo_operation in ["zoom_in", "zoom_out"]:
            center_x_ratio = float(self.geometric_params["zoom_x"]["entry"].get())
            center_y_ratio = float(self.geometric_params["zoom_y"]["entry"].get())
            factor = float(self.geometric_params["zoom_scale"]["entry"].get())
            if factor <=0:
                messagebox.showerror("Hata", "Zoom ölçeği pozitif olmalıdır.")
                return

            orig_w, orig_h = self.original_image_size
            center_x_px = int(center_x_ratio * orig_w)
            center_y_px = int(center_y_ratio * orig_h)
            
            temp_processed_data = self.zoom(self.original_image_data, factor, 
                                          center_x_px, center_y_px, interp_method, 
                                          geo_operation == "zoom_in")
                                              
        elif geo_operation == "rotate":
            angle = float(self.geometric_params["angle"]["entry"].get())
            temp_processed_data = self.rotate_image(self.original_image_data, angle, interp_method)

        if temp_processed_data:
            self.processed_image_data = temp_processed_data
            self.display_image(self.processed_image_data, self.processed_label)
        else:
            messagebox.showinfo("Bilgi", "Geometrik işlem bir sonuç üretmedi veya iptal edildi.")


    def apply_scurve_operation(self):
        """S-Curve kontrast güçlendirme uygula"""
        if self.current_image_cv is None:
            messagebox.showerror("Hata", "S-Curve işlemi için görüntü gerekli (OpenCV).")
            return

        scurve_type = self.scurve_type_var.get()
        slope_a = float(self.slope_entry.get()) # 'a' in formulas
        shift_b = float(self.shift_entry.get()) # 'b' in formulas, or center adjustment

        img_normalized = self.current_image_cv.astype(np.float32) / 255.0
        
        processed_normalized = np.zeros_like(img_normalized)

        if scurve_type == "standard":
            processed_normalized = 1 / (1 + np.exp(-slope_a * (img_normalized - 0.5))) 
        elif scurve_type == "shifted":
            processed_normalized = 1 / (1 + np.exp(-slope_a * (img_normalized - shift_b)))
        elif scurve_type == "slope_based": # Renamed from "slope"
             processed_normalized = 1 / (1 + np.exp(-slope_a * (img_normalized - 0.5))) # Re-using standard centered sigmoid
        elif scurve_type == "custom_tanh":
            processed_normalized = (np.tanh(slope_a * (img_normalized - 0.5)) * 0.5) + 0.5

        processed_cv = np.clip(processed_normalized * 255, 0, 255).astype(np.uint8)
        self.convert_cv_to_txt(processed_cv)


    def apply_hough_operation(self):
        """Hough Transform uygula"""
        if self.current_image_cv is None:
            messagebox.showerror("Hata", "Hough Transform için görüntü gerekli (OpenCV).")
            return

        hough_type = self.hough_type_var.get()
        
        if hough_type == "lines":
            self.detect_lines()
        elif hough_type == "eyes":
            self.detect_eyes()

    def detect_lines(self):
        """Çizgi tespiti"""
        threshold = int(self.hough_threshold_entry.get())
        min_line_length = int(self.min_line_length_entry.get())
        max_line_gap = int(self.max_line_gap_entry.get()) # Get max_line_gap
        
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 
                                minLineLength=min_line_length, maxLineGap=max_line_gap) # Use max_line_gap
        
        result_cv = self.current_image_cv.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_cv, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green lines
        
        self.convert_cv_to_txt(result_cv)

    def detect_eyes(self):
        """Göz tespiti"""
        try:
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml' 
           
            if not cv2.os.path.exists(eye_cascade_path):
                 eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                 if not cv2.os.path.exists(eye_cascade_path):
                     messagebox.showerror("Hata", "Haarcascade göz dosyası bulunamadı. OpenCV kurulumunu kontrol edin.")
                     return

            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            if eye_cascade.empty():
                messagebox.showerror("Hata", "Haarcascade göz sınıflandırıcısı yüklenemedi.")
                return

            gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_RGB2GRAY)
           
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20)) 
            
            result_cv = self.current_image_cv.copy()
            for (x, y, w, h) in eyes:
                cv2.rectangle(result_cv, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue rectangles for eyes
               
            self.convert_cv_to_txt(result_cv)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Göz tespiti hatası: {e}")


    def apply_deblur_operation(self):
        """Deblurring uygula"""
        if self.current_image_cv is None:
            messagebox.showerror("Hata", "Deblurring için görüntü gerekli (OpenCV).")
            return

        try:
            kernel_size = int(self.kernel_size_entry.get())
            if kernel_size <= 0 or kernel_size % 2 == 0:
                messagebox.showerror("Hata", "Kernel boyutu pozitif tek sayı olmalıdır.")
                return
            angle = float(self.blur_angle_entry.get())
            
            psf = self.create_motion_blur_kernel(kernel_size, angle)
            
            
            img_float = self.current_image_cv.astype(np.float32) / 255.0
            
            deblurred_channels = []
            for i in range(img_float.shape[2]): # Iterate R, G, B
                channel = img_float[:,:,i]
                
                channel_fft = np.fft.fft2(channel)
                psf_fft = np.fft.fft2(psf, s=channel.shape) 
                K = 0.01 # Noise factor, needs tuning
                wiener_filter_fft = np.conj(psf_fft) / (np.abs(psf_fft)**2 + K)
                
                deblurred_channel_fft = channel_fft * wiener_filter_fft
                deblurred_channel = np.fft.ifft2(deblurred_channel_fft)
                deblurred_channels.append(np.abs(deblurred_channel)) # Take absolute value

            deblurred_cv_float = np.stack(deblurred_channels, axis=-1)
            deblurred_cv = np.clip(deblurred_cv_float * 255, 0, 255).astype(np.uint8)
            
            self.convert_cv_to_txt(deblurred_cv)
        except Exception as e:
            messagebox.showerror("Hata", f"Deblurring sırasında hata: {e}")
            import traceback
            print(f"Deblurring error: {e}\n{traceback.format_exc()}")


    def create_motion_blur_kernel(self, size, angle_degrees):
        """Motion blur kernel (Point Spread Function - PSF) oluştur"""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        angle_rad = np.deg2rad(angle_degrees)
        
      
        if abs(angle_degrees % 180) < 1e-3 : # Horizontal
             kernel[center, :] = 1
        elif abs((angle_degrees - 90) % 180) < 1e-3: # Vertical
             kernel[:, center] = 1
        else: 
            for i in range(size):
                x = center + (i - center) * math.cos(angle_rad)
                y = center + (i - center) * math.sin(angle_rad) # OpenCV y is downwards
                x_round, y_round = int(round(x)), int(round(y))
                if 0 <= x_round < size and 0 <= y_round < size:
                    kernel[y_round, x_round] = 1 # PSF is (rows, cols) -> (y,x)

        if np.sum(kernel) == 0: 
            kernel[center,center] = 1 
        kernel = kernel / np.sum(kernel)
        return kernel


    def apply_object_detection(self):
        """Nesne tespiti ve özellik çıkarma"""
        if self.current_image_cv is None:
            messagebox.showerror("Hata", "Nesne tespiti için görüntü gerekli (OpenCV).")
            return

        try:
            hsv_lower_str = self.hsv_lower_entry.get().split(',')
            hsv_upper_str = self.hsv_upper_entry.get().split(',')
            
            if len(hsv_lower_str) != 3 or len(hsv_upper_str) != 3:
                messagebox.showerror("Hata", "HSV sınırları 3 değer içermelidir (H,S,V).")
                return

            hsv_lower = np.array([int(x.strip()) for x in hsv_lower_str])
            hsv_upper = np.array([int(x.strip()) for x in hsv_upper_str])
            
            hsv_image = cv2.cvtColor(self.current_image_cv, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
            
            # Optional: Morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result_cv = self.current_image_cv.copy()
            self.object_features_data = [] # Reset features list

            gray_image_for_features = cv2.cvtColor(self.current_image_cv, cv2.COLOR_RGB2GRAY)

            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 50: # Filter small contours (noise)
                    continue

                M = cv2.moments(contour)
                center_x, center_y = -1, -1
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                
                rect = cv2.minAreaRect(contour) # ((center_x, center_y), (width, height), angle)
                (box_center_x, box_center_y), (box_width, box_height), angle_rect = rect
                length = max(box_width, box_height)
                width_feat = min(box_width, box_height) # 'width' is a class member, use 'width_feat'

                diagonal = math.sqrt(length**2 + width_feat**2)

               
                contour_mask = np.zeros(gray_image_for_features.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
                
                pixels_in_contour = gray_image_for_features[contour_mask == 255]
                
                mean_val, median_val, energy, entropy = -1, -1, -1, -1
                if pixels_in_contour.size > 0:
                    mean_val = np.mean(pixels_in_contour)
                    median_val = np.median(pixels_in_contour)
                    
                    # Histogram for Energy and Entropy
                    hist = cv2.calcHist([gray_image_for_features], [0], contour_mask, [256], [0, 256])
                    hist = hist.astype("float")
                    if np.sum(hist) > 0: # Ensure sum is not zero before normalizing
                        hist /= np.sum(hist) # Normalize
                        energy = np.sum(hist**2)
                        entropy = -np.sum(hist * np.log2(hist + 1e-7)) # Add epsilon for log(0)
                
                self.object_features_data.append({
                    "No": i + 1,
                    "Center": f"{center_x},{center_y}",
                    "Length (px)": f"{length:.2f}", # Using minAreaRect dimensions
                    "Width (px)": f"{width_feat:.2f}",
                    "Diagonal (px)": f"{diagonal:.2f}",
                    "Mean Intensity": f"{mean_val:.2f}",
                    "Median Intensity": f"{median_val:.2f}",
                    "Energy": f"{energy:.4f}",
                    "Entropy": f"{entropy:.2f}"
                })

                # Draw on result image
                cv2.drawContours(result_cv, [contour], -1, (0, 255, 0), 2) # Green contour
                if center_x != -1:
                    cv2.circle(result_cv, (center_x, center_y), 3, (0, 0, 255), -1) # Red center dot
                # Draw rotated bounding box
                box_points = cv2.boxPoints(rect)
                box_points = np.intp(box_points) # np.int0 is deprecated, use np.intp
                cv2.drawContours(result_cv, [box_points],0,(255,0,0),1) # Blue rotated box


            self.convert_cv_to_txt(result_cv)
            if not self.object_features_data:
                messagebox.showinfo("Bilgi", "Belirtilen HSV aralığında nesne bulunamadı.")
            else:
                 messagebox.showinfo("Başarılı", f"{len(self.object_features_data)} nesne bulundu. Özellikler Excel'e kaydedilmeye hazır.")

        except ValueError as ve: # Catch specific error for HSV input
            messagebox.showerror("Giriş Hatası", f"HSV değerleri hatalı: {ve}")
        except Exception as e:
            messagebox.showerror("Hata", f"Nesne tespiti sırasında hata: {e}")
            import traceback
            print(f"Object detection error: {e}\n{traceback.format_exc()}")


    def save_to_excel(self):
        """Tespit edilen nesne özelliklerini Excel'e kaydeder."""
        if not self.object_features_data:
            messagebox.showwarning("Uyarı", "Kaydedilecek nesne özelliği verisi yok. Lütfen önce nesne tespiti yapın.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filepath:
            try:
                df = pd.DataFrame(self.object_features_data)
                df.to_excel(filepath, index=False)
                messagebox.showinfo("Başarılı", f"Özellikler {filepath} dosyasına kaydedildi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Excel'e kaydedilirken hata: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
