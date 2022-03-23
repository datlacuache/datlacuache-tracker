# Datlacuache Object Tracker [WIP 🛠]

Repo oficial de _object tracking_ para Datlacuache, 2022. Más cotenido pronto.

Este es un trabajo en proceso. [WIP 🛠]

## Setup ⚙️

Si utilizas una CPU con Anaconda o Miniconda, puedes ejecutar el siguiente 
comando para crear tu entorno:

```bash
conda env create -f env_cpu.yml python=3.8
```

## Uso 🖼

Si deseas procesar un video y sólamente detectar objetos, puedes ejecutar el
script `detect_video.py` modificando las variables de configuración, como
`video` con la ruta del video y `tracked_classes` con las classes a seguir.

```bash
python detect_video.py
```

> Nota: Dado que en este punto no se ha admitido la integración de modelos
  _custom_, se utilizan las etiquetas de COCO y un modelo entrenado con
  dicho conjunto de datos.


### To Do 🚧

- [ ] Integrar el uso de modelos custom con la estructura utilizada en el 
proyecto original.
- [ ] Integrar el uso de un script o cuaderno para entrenamiento de modelos.
- [ ] Generalizar la definición de recta-dirección para el conteo de objetos.
