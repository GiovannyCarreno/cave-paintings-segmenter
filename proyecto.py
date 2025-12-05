import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

class SegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentación de Imágenes con IA")
        self.root.geometry("1200x800")
        
        # Variables
        self.model = None
        self.current_image = None
        self.image_path = ""
        self.model_path = "mejor_modelo_por_loss.h5"
        
        # Configurar la interfaz
        self.setup_ui()
        
        # Cargar modelo al inicio
        self.load_model()
    
    def dice_coefficient(self, y_true, y_pred, smooth=1):
        """Función de métrica Dice personalizada"""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar el grid para que se expanda
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Segmentación de Imágenes con IA", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Botón para seleccionar imagen
        self.select_btn = ttk.Button(control_frame, text="Seleccionar Imagen", 
                                    command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Label para mostrar la ruta seleccionada
        self.path_label = ttk.Label(control_frame, text="Ninguna imagen seleccionada")
        self.path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Botón para procesar
        self.process_btn = ttk.Button(control_frame, text="Procesar Imagen", 
                                     command=self.process_image, state="disabled")
        self.process_btn.grid(row=0, column=2, padx=(10, 0))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Frame para las imágenes
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Estado del modelo
        self.status_label = ttk.Label(main_frame, text="Cargando modelo...", 
                                     foreground="blue")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=(10, 0))
    
    def load_model(self):
        """Cargar el modelo en un hilo separado"""
        def load_model_thread():
            try:
                if os.path.exists(self.model_path):
                    self.model = load_model(self.model_path, 
                                          custom_objects={'dice_coefficient': self.dice_coefficient})
                    self.root.after(0, lambda: self.status_label.config(
                        text="Modelo cargado correctamente", foreground="green"))
                else:
                    self.root.after(0, lambda: self.status_label.config(
                        text="Error: No se encontró el archivo del modelo", foreground="red"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error al cargar el modelo: {str(e)}", foreground="red"))
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def select_image(self):
        """Seleccionar imagen desde el explorador de archivos"""
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen para segmentar",
            filetypes=filetypes
        )
        
        if filename:
            self.image_path = filename
            # Mostrar solo el nombre del archivo en la etiqueta
            display_name = os.path.basename(filename)
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."
            self.path_label.config(text=display_name)
            self.process_btn.config(state="normal")
    
    def overlay_mask(self, image, mask, color=(0, 255, 0), alpha=0.3):
        """Superponer máscara sobre imagen con color y transparencia"""
        overlay = image.copy()
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask == 1] = color
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay
    
    def process_image(self):
        """Procesar la imagen seleccionada"""
        if not self.model:
            messagebox.showerror("Error", "El modelo no está cargado")
            return
        
        if not self.image_path:
            messagebox.showerror("Error", "Por favor selecciona una imagen")
            return
        
        # Iniciar procesamiento en hilo separado
        def process_thread():
            try:
                self.root.after(0, lambda: self.progress.start())
                self.root.after(0, lambda: self.process_btn.config(state="disabled"))
                
                # Leer y procesar imagen
                img = cv2.imread(self.image_path)
                if img is None:
                    raise ValueError("No se pudo cargar la imagen")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensionar
                IMG_SIZE = 128
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_norm = img_resized / 255.0
                input_data = np.expand_dims(img_norm, axis=0)
                
                # Predecir
                pred_mask = self.model.predict(input_data)
                pred_mask = pred_mask[0, :, :, 0]
                mask_bin = (pred_mask > 0.5).astype(np.uint8)
                
                # Crear overlay
                overlay_1 = self.overlay_mask(img_resized, mask_bin, color=(0, 255, 0))  # Color verde
                
                # Mostrar resultados en el hilo principal
                self.root.after(0, lambda: self.display_results(img_resized, overlay_1))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error al procesar: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.process_btn.config(state="normal"))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def display_results(self, original_img, segmented_img):
        """Mostrar los resultados en la interfaz"""
        # Limpiar frame anterior
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Crear figura de matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagen original
        ax1.imshow(original_img)
        ax1.set_title("Imagen Original", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Imagen segmentada
        ax2.imshow(segmented_img)
        ax2.set_title("Máscara Superpuesta", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Integrar matplotlib con tkinter
        canvas = FigureCanvasTkAgg(fig, self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Guardar referencia para evitar que se elimine por el garbage collector
        self.current_canvas = canvas

def main():
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()