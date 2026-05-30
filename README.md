# jocarsa | entrenador IA

Versión corregida.

## Cambio principal

Ahora el editor JSONL carga automáticamente el dataset por defecto al arrancar.

Además, las rutas relativas se resuelven respecto a la carpeta donde está `jocarsa_entrenador_ia.py`, no respecto al directorio desde el que se lanza Python.

## Instalación

```bash
pip install ttkbootstrap psutil torch transformers datasets peft bitsandbytes accelerate
```

## Ejecución

```bash
python jocarsa_entrenador_ia.py
```
