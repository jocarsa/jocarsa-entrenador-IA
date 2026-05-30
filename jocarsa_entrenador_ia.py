#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jocarsa | entrenador IA

Aplicación Tkinter + ttkbootstrap para:
1. Probar un modelo base antes del fine-tuning.
2. Entrenar un adaptador QLoRA con un JSONL.
3. Probar el modelo después del fine-tuning.
4. Monitorizar CPU, GPU, RAM y VRAM.
5. Editar el dataset JSONL con un CRUD integrado.

Instalación:

pip install ttkbootstrap psutil torch transformers datasets peft bitsandbytes accelerate

Ejecución:

python jocarsa_entrenador_ia.py
"""

import json
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

try:
    import psutil
except Exception:
    psutil = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_NAME = "jose_vicente_carratala_entrenamiento.jsonl"
DEFAULT_DATASET_PATH = APP_DIR / DEFAULT_DATASET_NAME


DEFAULT_EXAMPLES = [
    {
        "text": "Usuario: ¿Quién es Jose Vicente Carratalá Sanchis?\nAsistente: Jose Vicente Carratalá Sanchis es programador, docente, autor técnico y creador de proyectos relacionados con programación, formación, software empresarial, inteligencia artificial y visualización 3D."
    },
    {
        "text": "Usuario: ¿Qué es Jocarsa?\nAsistente: Jocarsa es una marca utilizada por Jose Vicente Carratalá para proyectos de software, formación tecnológica, soluciones SaaS, herramientas de productividad y aplicaciones empresariales."
    },
    {
        "text": "Usuario: ¿Qué estilo de respuesta prefiere Jose Vicente Carratalá?\nAsistente: Prefiere respuestas completas, directas, técnicas, en tono neutral y no coloquial, normalmente con código completo cuando se trata de programación."
    },
    {
        "text": "Usuario: ¿Quién es Moni en los proyectos de Jose Vicente?\nAsistente: Moni es una asistente llamada Monitor de Recursos, diseñada para responder en español sobre informática, tecnología y educación, con integración visual en Blender."
    }
]


class ConsoleRedirector:
    """Redirige print() a la consola interna de Tkinter."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self):
        pass


def ensure_default_dataset():
    """Crea un dataset mínimo junto al script si no existe."""
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH

    with DEFAULT_DATASET_PATH.open("w", encoding="utf-8") as f:
        for item in DEFAULT_EXAMPLES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return DEFAULT_DATASET_PATH


def resolve_path(path_text):
    """
    Resuelve rutas relativas.

    Primero prueba la ruta tal cual.
    Si no existe y es relativa, prueba junto a la carpeta del script.
    """

    path = Path(path_text).expanduser()

    if path.exists():
        return path.resolve()

    if not path.is_absolute():
        candidate = APP_DIR / path
        if candidate.exists():
            return candidate.resolve()

    return path


def parse_text_record(text):
    """Extrae pregunta y respuesta desde el campo text."""

    text = text.strip()

    if "Usuario:" in text and "Asistente:" in text:
        before, after = text.split("Asistente:", 1)
        question = before.replace("Usuario:", "", 1).strip()
        answer = after.strip()
        return question, answer

    if "Pregunta:" in text and "Respuesta:" in text:
        before, after = text.split("Respuesta:", 1)
        question = before.replace("Pregunta:", "", 1).strip()
        answer = after.strip()
        return question, answer

    return "", text


def build_text_record(question, answer):
    """Construye el campo text para entrenamiento."""
    return f"Usuario: {question.strip()}\nAsistente: {answer.strip()}"


def load_jsonl(path):
    """Lee un JSONL con clave text."""
    path = resolve_path(path)

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "text" not in obj:
                raise ValueError(f"La línea {line_number} no contiene la clave 'text'.")

            question, answer = parse_text_record(obj["text"])

            records.append({
                "question": question,
                "answer": answer,
                "text": obj["text"]
            })

    return records


def save_jsonl(path, records):
    """Guarda registros en JSONL."""
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = APP_DIR / path

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            text = build_text_record(record["question"], record["answer"])
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def validate_jsonl(path):
    records = load_jsonl(path)

    if not records:
        raise ValueError("El dataset no contiene registros.")

    return len(records)


def read_gpu_stats():
    """Lee GPU/VRAM usando nvidia-smi."""

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1
        )

        if result.returncode != 0:
            return None

        line = result.stdout.strip().splitlines()[0]
        gpu_util, mem_used, mem_total = [float(x.strip()) for x in line.split(",")]

        vram_percent = 0.0
        if mem_total > 0:
            vram_percent = mem_used / mem_total * 100.0

        return {
            "gpu": gpu_util,
            "vram": vram_percent,
            "vram_used": mem_used,
            "vram_total": mem_total
        }

    except Exception:
        return None


class MiniChart(ttk.Frame):
    """Gráfica sencilla de línea para porcentajes."""

    def __init__(self, parent, title, max_points=90, **kwargs):
        super().__init__(parent, **kwargs)

        self.title = title
        self.max_points = max_points
        self.values = []

        self.columnconfigure(0, weight=1)

        self.label = ttk.Label(self, text=f"{title}: —", font=("Ubuntu", 10, "bold"))
        self.label.grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            self,
            height=74,
            background="#111827",
            highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def add_value(self, value, extra_text=""):
        if value is None:
            self.label.configure(text=f"{self.title}: no disponible")
            return

        value = max(0, min(100, float(value)))
        self.values.append(value)

        if len(self.values) > self.max_points:
            self.values = self.values[-self.max_points:]

        if extra_text:
            self.label.configure(text=f"{self.title}: {value:.1f}% · {extra_text}")
        else:
            self.label.configure(text=f"{self.title}: {value:.1f}%")

        self.redraw()

    def redraw(self):
        self.canvas.delete("all")

        w = max(self.canvas.winfo_width(), 10)
        h = max(self.canvas.winfo_height(), 10)

        for p in [25, 50, 75]:
            y = h - (p / 100.0) * h
            self.canvas.create_line(0, y, w, y, fill="#243244")

        if len(self.values) < 2:
            return

        step = w / max(1, self.max_points - 1)
        offset = self.max_points - len(self.values)

        points = []

        for i, value in enumerate(self.values):
            x = (i + offset) * step
            y = h - (value / 100.0) * h
            points.extend([x, y])

        self.canvas.create_line(points, fill="#22c55e", width=2, smooth=True)

        last_x = (len(self.values) - 1 + offset) * step
        last_y = h - (self.values[-1] / 100.0) * h
        self.canvas.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill="#bbf7d0", outline="")


class JocarsaEntrenadorIA(ttk.Window):

    def __init__(self):
        super().__init__(themename="flatly")

        self.title("jocarsa | entrenador IA")
        self.geometry("1360x850")
        self.minsize(1180, 760)

        self.log_queue = queue.Queue()
        self.busy = False
        self.dataset_records = []
        self.current_index = None

        ensure_default_dataset()

        self.create_vars()
        self.create_ui()

        self.after(100, self.consume_logs)
        self.after(300, self.auto_load_default_dataset)
        self.after(1000, self.update_monitor)

    def create_vars(self):
        self.model_var = tk.StringVar(value="Qwen/Qwen2.5-0.5B-Instruct")
        self.dataset_var = tk.StringVar(value=str(DEFAULT_DATASET_PATH))
        self.adapter_var = tk.StringVar(value=str(APP_DIR / "adaptador_jose_vicente"))
        self.output_var = tk.StringVar(value=str(APP_DIR / "salida_entrenamiento"))

        self.epochs_var = tk.IntVar(value=8)
        self.batch_var = tk.IntVar(value=1)
        self.grad_var = tk.IntVar(value=4)
        self.lr_var = tk.StringVar(value="2e-4")
        self.max_len_var = tk.IntVar(value=384)
        self.new_tokens_var = tk.IntVar(value=120)

        self.use_4bit_var = tk.BooleanVar(value=True)
        self.use_fp16_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Listo")

    def create_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=16, bootstyle="dark")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(
            header,
            text="jocarsa | entrenador IA",
            font=("Ubuntu", 24, "bold"),
            bootstyle="inverse-dark"
        ).pack(side=LEFT)

        ttk.Label(
            header,
            text="QLoRA · datasets JSONL · monitorización CPU/GPU",
            font=("Ubuntu", 11),
            bootstyle="inverse-dark"
        ).pack(side=LEFT, padx=22)

        left = ttk.Frame(self, padding=12)
        left.grid(row=1, column=0, sticky="ns")

        right = ttk.Frame(self, padding=12)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.create_left_panel(left)
        self.create_tabs(right)

        status = ttk.Frame(self, padding=(12, 4))
        status.grid(row=2, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(0, weight=1)

        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(status, mode="indeterminate", bootstyle="success-striped")
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0), ipadx=90)

    def create_left_panel(self, parent):
        config = ttk.Labelframe(parent, text="Configuración", padding=12)
        config.grid(row=0, column=0, sticky="new")
        config.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(config, text="Modelo").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.model_var, width=40).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(config, text="Dataset").grid(row=row, column=0, sticky="w", pady=4)
        ds_frame = ttk.Frame(config)
        ds_frame.grid(row=row, column=1, sticky="ew", pady=4)
        ds_frame.columnconfigure(0, weight=1)
        ttk.Entry(ds_frame, textvariable=self.dataset_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(ds_frame, text="...", width=3, command=self.pick_dataset).grid(row=0, column=1, padx=(5, 0))
        row += 1

        ttk.Label(config, text="Adaptador").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.adapter_var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(config, text="Salida").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.output_var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Separator(config).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        settings = [
            ("Épocas", self.epochs_var, 1, 100, 1),
            ("Batch", self.batch_var, 1, 16, 1),
            ("Grad accum", self.grad_var, 1, 64, 1),
            ("Max length", self.max_len_var, 64, 4096, 64),
            ("Max tokens", self.new_tokens_var, 16, 2048, 16),
        ]

        for label, var, start, end, inc in settings:
            ttk.Label(config, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Spinbox(config, from_=start, to=end, increment=inc, textvariable=var, width=9).grid(row=row, column=1, sticky="w", pady=4)
            row += 1

        ttk.Label(config, text="Learning rate").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.lr_var, width=10).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        ttk.Checkbutton(config, text="4 bits", variable=self.use_4bit_var, bootstyle="success-round-toggle").grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        row += 1

        ttk.Checkbutton(config, text="fp16", variable=self.use_fp16_var, bootstyle="success-round-toggle").grid(row=row, column=0, columnspan=2, sticky="w", pady=5)

        actions = ttk.Labelframe(parent, text="Acciones", padding=12)
        actions.grid(row=1, column=0, sticky="new", pady=12)
        actions.columnconfigure(0, weight=1)

        ttk.Button(actions, text="1 · Probar antes", bootstyle="secondary", command=self.test_before).grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Button(actions, text="2 · Entrenar QLoRA", bootstyle="primary", command=self.train).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(actions, text="3 · Probar después", bootstyle="success", command=self.test_after).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(actions, text="Cargar JSONL en editor", bootstyle="info-outline", command=self.load_dataset_editor).grid(row=3, column=0, sticky="ew", pady=(14, 4))
        ttk.Button(actions, text="Guardar JSONL editor", bootstyle="success-outline", command=self.save_dataset_editor).grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(actions, text="Limpiar consola", bootstyle="warning-outline", command=self.clear_console).grid(row=5, column=0, sticky="ew", pady=(14, 4))

    def create_tabs(self, parent):
        tabs = ttk.Notebook(parent)
        tabs.grid(row=0, column=0, sticky="nsew")

        tab_train = ttk.Frame(tabs, padding=10)
        tab_editor = ttk.Frame(tabs, padding=10)
        tab_monitor = ttk.Frame(tabs, padding=10)

        tabs.add(tab_train, text="Entrenamiento")
        tabs.add(tab_editor, text="Editor JSONL")
        tabs.add(tab_monitor, text="Monitor")

        self.create_training_tab(tab_train)
        self.create_editor_tab(tab_editor)
        self.create_monitor_tab(tab_monitor)

    def create_training_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        questions_box = ttk.Labelframe(parent, text="Preguntas de prueba", padding=10)
        questions_box.grid(row=0, column=0, sticky="ew")
        questions_box.columnconfigure(0, weight=1)

        self.questions_text = tk.Text(questions_box, height=6, wrap="word", font=("Ubuntu Mono", 10))
        self.questions_text.grid(row=0, column=0, sticky="ew")
        self.questions_text.insert(
            "1.0",
            "¿Quién es Jose Vicente Carratalá Sanchis?\n"
            "¿Qué es Jocarsa?\n"
            "¿Qué tipo de proyectos desarrolla Jose Vicente Carratalá?\n"
            "¿Qué estilo de respuesta prefiere Jose Vicente Carratalá?\n"
            "¿Quién es Moni en los proyectos de Jose Vicente?"
        )

        console_box = ttk.Labelframe(parent, text="Consola", padding=10)
        console_box.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        console_box.columnconfigure(0, weight=1)
        console_box.rowconfigure(0, weight=1)

        self.console = tk.Text(
            console_box,
            wrap="word",
            font=("Ubuntu Mono", 10),
            background="#101820",
            foreground="#f2f2f2",
            insertbackground="#ffffff"
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(console_box, command=self.console.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scroll.set)

    def create_editor_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = ttk.Labelframe(parent, text="Registros", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.records_list = tk.Listbox(left, font=("Ubuntu", 10))
        self.records_list.grid(row=0, column=0, sticky="nsew")
        self.records_list.bind("<<ListboxSelect>>", self.on_record_select)

        list_scroll = ttk.Scrollbar(left, command=self.records_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.records_list.configure(yscrollcommand=list_scroll.set)

        list_buttons = ttk.Frame(left)
        list_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        list_buttons.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(list_buttons, text="Nuevo", bootstyle="info", command=self.new_record).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(list_buttons, text="Actualizar", bootstyle="primary", command=self.update_record).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(list_buttons, text="Eliminar", bootstyle="danger", command=self.delete_record).grid(row=0, column=2, sticky="ew", padx=2)

        right = ttk.Labelframe(parent, text="Pregunta y respuesta", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=2)

        ttk.Label(right, text="Pregunta").grid(row=0, column=0, sticky="w")
        self.question_editor = tk.Text(right, height=5, wrap="word", font=("Ubuntu Mono", 10))
        self.question_editor.grid(row=1, column=0, sticky="nsew", pady=(4, 10))

        ttk.Label(right, text="Respuesta").grid(row=2, column=0, sticky="w")
        self.answer_editor = tk.Text(right, height=12, wrap="word", font=("Ubuntu Mono", 10))
        self.answer_editor.grid(row=3, column=0, sticky="nsew", pady=(4, 10))

        buttons = ttk.Frame(right)
        buttons.grid(row=4, column=0, sticky="ew")
        buttons.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(buttons, text="Cargar JSONL", bootstyle="secondary", command=self.load_dataset_editor).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(buttons, text="Guardar JSONL", bootstyle="success", command=self.save_dataset_editor).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(buttons, text="Duplicar", bootstyle="info-outline", command=self.duplicate_record).grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(buttons, text="Vaciar campos", bootstyle="warning-outline", command=self.clear_record_fields).grid(row=0, column=3, sticky="ew", padx=3)

    def create_monitor_tab(self, parent):
        parent.columnconfigure((0, 1), weight=1)

        self.cpu_chart = MiniChart(parent, "CPU")
        self.cpu_chart.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        self.ram_chart = MiniChart(parent, "RAM")
        self.ram_chart.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        self.gpu_chart = MiniChart(parent, "GPU")
        self.gpu_chart.grid(row=1, column=0, sticky="ew", padx=8, pady=8)

        self.vram_chart = MiniChart(parent, "VRAM")
        self.vram_chart.grid(row=1, column=1, sticky="ew", padx=8, pady=8)

        info = ttk.Labelframe(parent, text="Información", padding=12)
        info.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(16, 8))
        info.columnconfigure(0, weight=1)

        msg = (
            "CPU y RAM se obtienen con psutil.\n"
            "GPU y VRAM se obtienen con nvidia-smi si hay una GPU NVIDIA disponible.\n"
            "Durante el entrenamiento, esta pestaña permite observar si realmente se está usando la GPU."
        )

        ttk.Label(info, text=msg, justify="left").grid(row=0, column=0, sticky="w")

    def auto_load_default_dataset(self):
        try:
            self.load_dataset_editor(show_message=False)
        except Exception as e:
            self.status_var.set(f"No se pudo cargar el JSONL inicial: {e}")

    def consume_logs(self):
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.console.insert("end", text)
                self.console.see("end")
        except queue.Empty:
            pass

        self.after(100, self.consume_logs)

    def run_thread(self, fn):
        if self.busy:
            messagebox.showwarning("Proceso activo", "Ya hay un proceso en marcha.")
            return

        self.busy = True
        self.status_var.set("Ejecutando proceso...")
        self.progress.start(10)

        def wrapper():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = ConsoleRedirector(self.log_queue)
            sys.stderr = ConsoleRedirector(self.log_queue)

            try:
                fn()
            except Exception:
                print("\nERROR:\n")
                print(traceback.format_exc())
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                self.after(0, self.finish_thread)

        threading.Thread(target=wrapper, daemon=True).start()

    def finish_thread(self):
        self.busy = False
        self.progress.stop()
        self.status_var.set("Listo")

    def clear_console(self):
        self.console.delete("1.0", "end")

    def update_monitor(self):
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            self.cpu_chart.add_value(cpu)
            self.ram_chart.add_value(ram.percent, f"{ram.used / 1024**3:.1f}/{ram.total / 1024**3:.1f} GB")
        else:
            self.cpu_chart.add_value(None)
            self.ram_chart.add_value(None)

        gpu = read_gpu_stats()

        if gpu:
            self.gpu_chart.add_value(gpu["gpu"])
            self.vram_chart.add_value(
                gpu["vram"],
                f"{gpu['vram_used']:.0f}/{gpu['vram_total']:.0f} MB"
            )
        else:
            self.gpu_chart.add_value(None)
            self.vram_chart.add_value(None)

        self.after(1000, self.update_monitor)

    def pick_dataset(self):
        path = filedialog.askopenfilename(
            title="Seleccionar dataset JSONL",
            initialdir=str(APP_DIR),
            filetypes=[("JSONL", "*.jsonl"), ("Todos", "*.*")]
        )

        if path:
            self.dataset_var.set(path)
            self.load_dataset_editor(show_message=False)

    def load_dataset_editor(self, show_message=True):
        path = resolve_path(self.dataset_var.get().strip())

        if not path.exists():
            path = ensure_default_dataset()

        self.dataset_var.set(str(path))
        self.dataset_records = load_jsonl(path)
        self.current_index = None
        self.refresh_records_list()
        self.clear_record_fields()

        self.status_var.set(f"Dataset cargado: {len(self.dataset_records)} registros")

        if show_message:
            messagebox.showinfo("Dataset cargado", f"Registros cargados: {len(self.dataset_records)}")

    def save_dataset_editor(self):
        path = self.dataset_var.get().strip()

        try:
            save_jsonl(path, self.dataset_records)
            self.status_var.set(f"Dataset guardado: {path}")
            messagebox.showinfo("Guardado", f"Dataset guardado correctamente:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al guardar JSONL", str(e))

    def refresh_records_list(self):
        self.records_list.delete(0, "end")

        for i, record in enumerate(self.dataset_records):
            q = record.get("question", "").strip()
            a = record.get("answer", "").strip()

            if q:
                label = f"{i + 1:04d} · {q[:80]}"
            else:
                label = f"{i + 1:04d} · {a[:80]}"

            self.records_list.insert("end", label)

    def on_record_select(self, event=None):
        sel = self.records_list.curselection()

        if not sel:
            return

        self.current_index = sel[0]
        record = self.dataset_records[self.current_index]

        self.question_editor.delete("1.0", "end")
        self.question_editor.insert("1.0", record.get("question", ""))

        self.answer_editor.delete("1.0", "end")
        self.answer_editor.insert("1.0", record.get("answer", ""))

    def get_editor_values(self):
        question = self.question_editor.get("1.0", "end").strip()
        answer = self.answer_editor.get("1.0", "end").strip()

        if not question:
            raise ValueError("La pregunta no puede estar vacía.")

        if not answer:
            raise ValueError("La respuesta no puede estar vacía.")

        return {
            "question": question,
            "answer": answer,
            "text": build_text_record(question, answer)
        }

    def new_record(self):
        try:
            record = self.get_editor_values()
            self.dataset_records.append(record)
            self.current_index = len(self.dataset_records) - 1
            self.refresh_records_list()
            self.records_list.selection_clear(0, "end")
            self.records_list.selection_set(self.current_index)
            self.records_list.see(self.current_index)
            self.status_var.set("Registro añadido")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_record(self):
        if self.current_index is None:
            messagebox.showwarning("Sin selección", "Selecciona un registro para actualizarlo.")
            return

        try:
            self.dataset_records[self.current_index] = self.get_editor_values()
            self.refresh_records_list()
            self.records_list.selection_set(self.current_index)
            self.records_list.see(self.current_index)
            self.status_var.set("Registro actualizado")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def delete_record(self):
        if self.current_index is None:
            messagebox.showwarning("Sin selección", "Selecciona un registro para eliminarlo.")
            return

        if not messagebox.askyesno("Eliminar", "¿Eliminar este registro del dataset?"):
            return

        del self.dataset_records[self.current_index]
        self.current_index = None
        self.refresh_records_list()
        self.clear_record_fields()
        self.status_var.set("Registro eliminado")

    def duplicate_record(self):
        if self.current_index is None:
            messagebox.showwarning("Sin selección", "Selecciona un registro para duplicarlo.")
            return

        record = dict(self.dataset_records[self.current_index])
        record["question"] = record.get("question", "") + " "
        self.dataset_records.insert(self.current_index + 1, record)
        self.current_index += 1
        self.refresh_records_list()
        self.records_list.selection_set(self.current_index)
        self.records_list.see(self.current_index)
        self.on_record_select()
        self.status_var.set("Registro duplicado")

    def clear_record_fields(self):
        self.question_editor.delete("1.0", "end")
        self.answer_editor.delete("1.0", "end")

    def get_questions(self):
        text = self.questions_text.get("1.0", "end").strip()
        return [line.strip() for line in text.splitlines() if line.strip()]

    def test_before(self):
        self.run_thread(self._test_before)

    def train(self):
        self.run_thread(self._train)

    def test_after(self):
        self.run_thread(self._test_after)

    def _load_base_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_name = self.model_var.get().strip()

        print(f"Cargando tokenizador: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Cargando modelo base...")

        if self.use_4bit_var.get():
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=qconf,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )

        model.eval()
        return model, tokenizer

    def _ask_questions(self, model, tokenizer):
        import torch

        questions = self.get_questions()

        if not questions:
            print("No hay preguntas de prueba.")
            return

        for question in questions:
            prompt = (
                "Responde de forma breve y precisa.\n\n"
                f"Usuario: {question}\n"
                "Asistente:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(self.new_tokens_var.get()),
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Asistente:" in answer:
                answer = answer.split("Asistente:", 1)[1].strip()

            print("\nPREGUNTA:")
            print(question)
            print("\nRESPUESTA:")
            print(answer)
            print("\n" + "-" * 70)

    def _test_before(self):
        print("\n============================================")
        print("PRUEBA ANTES DEL ENTRENAMIENTO")
        print("============================================\n")
        model, tokenizer = self._load_base_model()
        self._ask_questions(model, tokenizer)

    def _test_after(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print("\n============================================")
        print("PRUEBA DESPUÉS DEL ENTRENAMIENTO")
        print("============================================\n")

        model_name = self.model_var.get().strip()
        adapter_dir = resolve_path(self.adapter_var.get().strip())

        if not adapter_dir.exists():
            raise FileNotFoundError(f"No existe el adaptador: {adapter_dir}")

        print("Cargando tokenizador del adaptador...")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Cargando modelo base...")

        if self.use_4bit_var.get():
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=qconf,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )

        print("Cargando adaptador QLoRA...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        self._ask_questions(model, tokenizer)

    def _train(self):
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("\n============================================")
        print("ENTRENAMIENTO QLORA")
        print("============================================\n")

        model_name = self.model_var.get().strip()
        dataset_path = resolve_path(self.dataset_var.get().strip())
        adapter_dir = self.adapter_var.get().strip()
        output_dir = self.output_var.get().strip()

        total = validate_jsonl(dataset_path)
        print(f"Dataset válido: {dataset_path}")
        print(f"Ejemplos: {total}\n")

        print("Cargando dataset...")
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        print("Cargando tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Cargando modelo...")

        if self.use_4bit_var.get():
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=qconf,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )

        print("Preparando entrenamiento k-bit...")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        model = get_peft_model(model, lora_config)

        print("\nParámetros entrenables:")
        model.print_trainable_parameters()

        max_len = int(self.max_len_var.get())

        def tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )

        print("\nTokenizando...")
        tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(self.epochs_var.get()),
            per_device_train_batch_size=int(self.batch_var.get()),
            gradient_accumulation_steps=int(self.grad_var.get()),
            learning_rate=float(self.lr_var.get()),
            fp16=bool(self.use_fp16_var.get()),
            logging_steps=1,
            save_steps=50,
            report_to="none",
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=collator
        )

        print("\nEntrenando...\n")
        trainer.train()

        print("\nGuardando adaptador...")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        print("\nEntrenamiento terminado correctamente.")
        print(f"Adaptador guardado en: {adapter_dir}")


if __name__ == "__main__":
    app = JocarsaEntrenadorIA()
    app.mainloop()
