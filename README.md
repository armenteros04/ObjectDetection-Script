<img src="https://upload.wikimedia.org/wikipedia/commons/f/f2/Ankara_universitesi.png" width="100" height="100" />


# Detección de objetos en tiempo real
## Proyecto de Pattern Recognition
### Realizado por Antonio Manuel Armenteros Iranzo

En este proyecto realizado para la asignatura Pattern Recognition durante mi Erasmus en Ankara Üniversitesi, he implementado un sistema de detección de objetos en tiempo real usando el modelo **YOLOv8n** de Ultralytics. El sistema incluye seguimiento multiobjetivo mediante el algoritmo **BotSORT**, un contador de cruce de línea virtual configurable y registro en tiempo real por terminal.

## **Información importante**

Para ejecutar el sistema de detección basta con poner en una terminal lo siguiente:
```
python main.py
```

Por defecto el script usa el vídeo `videos/car.mp4`. Para cambiar a la webcam, comenta/descomenta estas líneas al inicio del script:
```python
cap = cv2.VideoCapture(1)              # ← webcam
# cap = cv2.VideoCapture("videos/car.mp4")  # ← fichero de vídeo
```

Para cambiar la posición de la línea de conteo, modifica el valor de `line_y` al inicio del script:
```python
line_y = 400   # coordenada Y en píxeles de la línea virtual
```

Para salir de la ventana de visualización en cualquier momento pulsa `q`.

Para instalar todas las dependencias necesarias ejecuta lo siguiente:
```
pip install ultralytics opencv-python torch torchvision
```

## **Fichero main.py**

En este fichero se encuentra el código principal del sistema. El flujo de procesamiento es el siguiente:

1. Captura de frames desde webcam o fichero de vídeo.
2. Llamada a `model.track()` con BotSORT para obtener detecciones enriquecidas con IDs persistentes.
3. Cálculo del centro `(cx, cy)` de cada bounding box.
4. Comparación de `cy` con el umbral `line_y`: si el objeto cruza la línea y su ID no ha sido contado, se incrementa el contador global.
5. Dibujado de bounding boxes, etiquetas, línea virtual y contadores sobre el frame.
6. Salida por terminal del estado en tiempo real mediante sobreescritura con retorno de carro.

**Pipeline de detección (YOLOv8n)**
```
Frame de entrada
      ↓
Backbone CSPDarknet  →  extrae mapas de características a 3 escalas
      ↓
Cuello PAN-FPN       →  fusiona características multi-escala
      ↓
Cabeza anchor-free   →  predicciones de caja + clase
      ↓
Post-procesado NMS   →  filtra duplicados (IoU ≥ 0.45, confianza ≥ 0.25)
```

**Pipeline de seguimiento (BotSORT)**
```
Predicción Kalman     →  estima posición del track en el frame actual
Compensación CMC      →  corrige movimiento de cámara con optical flow
Asignación húngara    →  empareja detecciones con tracks existentes
                          (coste = IoU espacial + similitud coseno ReID 512-dim)
Gestión de tracks     →  nuevo → activo → perdido → eliminado
```

**Pseudocódigo contador de cruce de línea**
```
para cada detección en el frame:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    si cy > line_y Y track_id NO está en already_counted:
        counter += 1
        already_counted.add(track_id)
        imprimir aviso de cruce por terminal
```

> ⚠️ El contador actual es **unidireccional** (de arriba a abajo). Un contador bidireccional que detecte entradas y salidas es una extensión natural para trabajo futuro.

## **Previsualización**



## **Resultados experimentales**

Rendimiento en FPS medido en el hardware de prueba:

| Configuración | Entrada | FPS medio | Notas |
|---|---|---|---|
| Detección + BotSORT | Webcam | ~15 | Limitado por hardware USB |
| Detección + BotSORT | car.mp4 | ~20 | Estable, acelerado por GPU |

Precisión del modelo YOLOv8n evaluado en COCO val2017:

| Métrica | Valor |
|---|---|
| mAP@0.5 | 0.38 |
| Precisión | 0.65 |
| Recall | 0.56 |
| Tamaño del modelo | 3.2M parámetros |

Hardware utilizado: Intel Core i5-11400H · NVIDIA RTX 3050 Laptop 4GB · 16 GB RAM · Windows 10

## **Stack tecnológico**

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3.13.2 |
| IDE | PyCharm 2024.3.2 |
| Modelo de detección | YOLOv8n (Ultralytics) |
| Algoritmo de tracking | BotSORT |
| Framework deep learning | PyTorch |
| Procesado de vídeo | OpenCV |
| Computación numérica | NumPy |
