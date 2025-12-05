# Segmentación de Pictogramas de la Región Andina de Colombia mediante Redes Neuronales

## Descripción

Este proyecto implementa un sistema de segmentación de imágenes utilizando redes neuronales profundas para identificar y segmentar pictogramas característicos de la región andina de Colombia. La aplicación cuenta con una interfaz gráfica intuitiva que permite cargar imágenes y visualizar los resultados de la segmentación en tiempo real.

## Características

- **Interfaz gráfica amigable**: Aplicación de escritorio desarrollada con Tkinter
- **Segmentación automática**: Procesamiento de imágenes mediante modelo de deep learning pre-entrenado
- **Visualización en tiempo real**: Comparación lado a lado de la imagen original y la máscara segmentada
- **Soporte múltiples formatos**: Compatible con JPG, JPEG, PNG, BMP, TIFF
- **Procesamiento asíncrono**: Carga y procesamiento en hilos separados para mantener la interfaz responsive

## Tecnologías Utilizadas

- **Python 3.x**
- **TensorFlow/Keras**: Framework de deep learning para el modelo de segmentación
- **OpenCV (cv2)**: Procesamiento de imágenes
- **Tkinter**: Interfaz gráfica de usuario
- **Matplotlib**: Visualización de resultados
- **NumPy**: Operaciones numéricas y manipulación de arrays

## Requisitos

### Dependencias

El proyecto requiere las siguientes librerías de Python:

```
tensorflow>=2.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.19.0
```

### Sistema Operativo

- Windows 10/11
- Linux
- macOS

## Instalación

1. **Clonar o descargar el repositorio**

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**
   - Windows:
     ```bash
     venv\Scripts\Activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias**
   ```bash
   pip install requirements.txt
   ```

## Estructura del Proyecto

```
Proyecto/
│
├── proyecto.py                    # Aplicación principal con interfaz gráfica
├── mejor_modelo_por_loss.h5      # Modelo de red neuronal pre-entrenado
├── README.md                      # Este archivo
│
└── imagenes de prueba/           # Carpeta con imágenes de ejemplo
    ├── imagen_15.jpeg
    ├── imagen_20.jpeg
    ├── imagen_21.jpg
    └── ...
```

## Uso

1. **Ejecutar la aplicación**
   ```bash
   python proyecto.py
   ```

2. **Usar la interfaz**:
   - Hacer clic en "Seleccionar Imagen" para cargar una imagen desde tu computadora
   - Una vez seleccionada la imagen, hacer clic en "Procesar Imagen"
   - La aplicación mostrará:
     - **Imagen Original**: La imagen cargada
     - **Máscara Superpuesta**: La imagen con la segmentación aplicada (en verde)

3. **Resultados**:
   - La máscara de segmentación se superpone sobre la imagen original con un color verde semitransparente
   - Las áreas segmentadas corresponden a los pictogramas identificados por el modelo

## Modelo de Red Neuronal

El proyecto utiliza un modelo de red neuronal convolucional (U-Net) entrenado específicamente para la segmentación de pictogramas andinos. El modelo:

- Utiliza la métrica **Dice Coefficient** para evaluar el rendimiento
- Procesa imágenes de tamaño 128x128 píxeles
- Genera máscaras binarias que identifican las regiones de interés

### Archivo del Modelo

El modelo pre-entrenado se encuentra en `mejor_modelo_por_loss.h5`. Asegúrate de que este archivo esté en el mismo directorio que `proyecto.py`.

## Configuración

### Color de la Máscara

El color de superposición de la máscara se puede modificar en la línea 170:

```python
overlay_1 = self.overlay_mask(img_resized, mask_bin, color=(0, 255, 0))  # RGB: Verde
```

## Solución de Problemas

### Error: "No se encontró el archivo del modelo"
- Asegúrate de que `mejor_modelo_por_loss.h5` esté en el mismo directorio que `proyecto.py`

### Error al cargar TensorFlow
- Verifica que TensorFlow esté correctamente instalado: `pip install tensorflow`

### La interfaz no responde durante el procesamiento
- El procesamiento se realiza en un hilo separado. Si la interfaz se congela, verifica que no haya errores en la consola.

## Notas

- Las imágenes se redimensionan automáticamente a 128x128 píxeles para el procesamiento
- El modelo utiliza un umbral de 0.5 para binarizar las predicciones
- La transparencia de la máscara superpuesta es del 30% (alpha=0.3)

## Contribuciones

Este proyecto fue desarrollado como parte de una electiva académica. Las contribuciones y mejoras son bienvenidas.

## Licencia

Este proyecto es de uso académico. Consulta con los autores para más información sobre el uso y distribución.

## Contacto

Para preguntas o sugerencias sobre este proyecto, por favor contacta a los desarrolladores.

Giovanny Carreño: carrenoestupinanh@gmail.com

---

**Desarrollado para la segmentación de pictogramas de la región andina de Colombia**

