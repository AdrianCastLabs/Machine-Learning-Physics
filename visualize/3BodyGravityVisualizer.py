import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "predictions" / "3body_predictions.csv"

BG = "#15191e"
PANEL = "#1d232b"
FIELD = "#252d37"
TEXT = "#dce3eb"
MUTED = "#93a0af"
GRID = "#303944"
COLORS = ("#7fa6ce", "#c5a16b", "#75aaa4")
BODY_NAMES = ("Body 01", "Body 02", "Body 03")


def load_data(path):
    required = [name for i in range(1, 4) for name in (f"x{i}", f"y{i}")]
    velocity_columns = [
        name for i in range(1, 4) for name in (f"vx{i}", f"vy{i}")
    ]
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        rows = list(reader)
        if not rows:
            raise ValueError("The CSV contains no samples.")

        def extract(columns):
            result = np.asarray(
                [[float(row[name]) for name in columns] for row in rows],
                dtype=float,
            )
            if not np.isfinite(result).all():
                raise ValueError("The CSV contains non-finite numeric values.")
            return result.reshape(-1, 3, 2)

        positions = extract(required)
        velocities = (
            extract(velocity_columns)
            if all(name in headers for name in velocity_columns)
            else None
        )
    return positions, velocities, headers


class Foldout(ttk.Frame):
    def __init__(self, parent, title, expanded=True):
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self.button = ttk.Button(self, command=self.toggle, style="Section.TButton")
        self.button.pack(fill="x")
        self.content = ttk.Frame(self, padding=(12, 8, 12, 14))
        self.refresh()

    def toggle(self):
        self.expanded = not self.expanded
        self.refresh()

    def refresh(self):
        self.button.configure(
            text=f"{'−' if self.expanded else '+'}  {self.title}"
        )
        if self.expanded:
            self.content.pack(fill="x")
        else:
            self.content.pack_forget()


class GravityViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Three-body / Trajectory laboratory")
        self.root.geometry("1380x900")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)

        self.positions = None
        self.velocities = None
        self.frame = 0
        self.playing = False
        self.syncing = False
        self.last_sidebar = ""
        self.speed = tk.StringVar(value="30")
        self.history = tk.StringVar(value="250")
        self.show_grid = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.follow = tk.BooleanVar(value=False)
        self.loop = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open a prediction CSV to begin.")
        self.sample_label = tk.StringVar(value="No dataset")
        self.source_label = tk.StringVar(value="—")
        self.coordinate_labels = []
        self.velocity_labels = []

        self.configure_style()
        self.build_interface()

        self.root.bind("<space>", self.toggle_play)
        self.root.bind("<Left>", lambda event: self.step(-1))
        self.root.bind("<Right>", lambda event: self.step(1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if DEFAULT_PATH.exists():
            self.open_path(DEFAULT_PATH)
        self.timer = self.root.after(33, self.tick)

    def configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            ".", background=PANEL, foreground=TEXT,
            font=("Segoe UI", 10), borderwidth=0,
        )
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure(
            "Title.TLabel", font=("Segoe UI", 19, "bold"),
            background=BG,
        )
        style.configure("Header.TFrame", background=BG)
        style.configure("Header.TLabel", background=BG, foreground=MUTED)
        style.configure("TButton", padding=(12, 7), background=FIELD)
        style.map("TButton", background=[("active", "#354252")])
        style.configure(
            "Section.TButton", anchor="w", padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TNotebook", background=BG, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", padding=(20, 10), background=PANEL)
        style.map(
            "TNotebook.Tab",
            background=[("selected", FIELD)],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "TCombobox", fieldbackground=FIELD, background=FIELD,
            foreground=TEXT, arrowcolor=TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD)],
            foreground=[("readonly", TEXT)],
        )
        style.configure("TCheckbutton", background=PANEL)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure(
            "Treeview", background=BG, fieldbackground=BG,
            foreground=TEXT, rowheight=28,
        )
        style.configure("Treeview.Heading", background=FIELD, padding=8)
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=FIELD)

    def build_interface(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(
            header, text="THREE-BODY", style="Title.TLabel"
        ).pack(side="left")
        ttk.Label(
            header, text="  /  TRAJECTORY LABORATORY", style="Header.TLabel"
        ).pack(side="left", padx=12)
        ttk.Button(header, text="Open CSV…", command=self.choose_file).pack(
            side="right"
        )

        workspace = ttk.Panedwindow(self.root, orient="horizontal")
        workspace.pack(fill="both", expand=True, padx=18)

        main = ttk.Frame(workspace)
        inspector = ttk.Frame(workspace, width=290)
        workspace.add(main, weight=5)
        workspace.add(inspector, weight=1)

        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill="both", expand=True)
        orbit = ttk.Frame(self.tabs)
        diagnostics = ttk.Frame(self.tabs)
        data = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(orbit, text="Orbit")
        self.tabs.add(diagnostics, text="Diagnostics")
        self.tabs.add(data, text="Data")

        self.figure = Figure(facecolor=BG)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=orbit)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, orbit, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.figure.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.94)

        self.diagnostic_figure = Figure(facecolor=BG)
        self.distance_ax, self.speed_ax = self.diagnostic_figure.subplots(2, 1)
        self.diagnostic_figure.subplots_adjust(
            left=0.10, right=0.96, bottom=0.10, top=0.93, hspace=0.48
        )
        self.diagnostic_canvas = FigureCanvasTkAgg(
            self.diagnostic_figure, master=diagnostics
        )
        self.diagnostic_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda event: self.render())

        ttk.Label(
            data,
            text="Current sample · positions and source velocities",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 12))
        columns = ("body", "x", "y", "vx", "vy", "speed")
        self.table = ttk.Treeview(
            data, columns=columns, show="headings", height=5
        )
        for name in columns:
            self.table.heading(name, text=name.upper())
            self.table.column(name, width=105, anchor="e", stretch=True)
        for i, name in enumerate(BODY_NAMES):
            self.table.insert("", "end", iid=str(i), values=(name,))
        self.table.pack(fill="x")
        ttk.Label(
            data,
            text=(
                "Units are inherited from the CSV. Frame index is not physical time.\n"
                "Speed uses source vx/vy columns; missing velocities are not estimated.\n"
                "Energy and momentum are omitted because masses and physical "
                "constants are not supplied."
            ),
            style="Muted.TLabel", wraplength=670, justify="left",
        ).pack(anchor="w", pady=20)

        # A scrollable inspector keeps every section accessible on small windows.
        inspector_canvas = tk.Canvas(
            inspector, bg=PANEL, highlightthickness=0, width=285
        )
        scrollbar = ttk.Scrollbar(
            inspector, orient="vertical", command=inspector_canvas.yview
        )
        scrollbar.pack(side="right", fill="y")
        inspector_canvas.pack(side="left", fill="both", expand=True)
        inspector_canvas.configure(yscrollcommand=scrollbar.set)
        sections = ttk.Frame(inspector_canvas)
        window = inspector_canvas.create_window(
            (0, 0), window=sections, anchor="nw"
        )
        sections.bind(
            "<Configure>",
            lambda event: inspector_canvas.configure(
                scrollregion=inspector_canvas.bbox("all")
            ),
        )
        inspector_canvas.bind(
            "<Configure>",
            lambda event: inspector_canvas.itemconfigure(window, width=event.width),
        )

        dataset = self.section(sections, "DATASET")
        ttk.Label(
            dataset, textvariable=self.source_label,
            wraplength=245, style="Muted.TLabel",
        ).pack(anchor="w")
        ttk.Label(dataset, textvariable=self.sample_label).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            dataset, text="Prediction samples · not a live integrator",
            wraplength=240, style="Muted.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        coordinates = self.section(sections, "POSITION")
        velocities = self.section(sections, "VELOCITY", expanded=False)
        for i, name in enumerate(BODY_NAMES):
            for parent, target in (
                (coordinates, self.coordinate_labels),
                (velocities, self.velocity_labels),
            ):
                ttk.Label(parent, text=name, foreground=COLORS[i]).pack(anchor="w")
                variable = tk.StringVar(value="—")
                target.append(variable)
                ttk.Label(
                    parent, textvariable=variable,
                    font=("Consolas", 10), wraplength=245,
                ).pack(anchor="w", pady=(3, 12))

        display = self.section(sections, "DISPLAY")
        for text, variable in (
            ("Coordinate grid", self.show_grid),
            ("Body labels", self.show_labels),
            ("Fit visible trajectories", self.follow),
        ):
            ttk.Checkbutton(
                display, text=text, variable=variable, command=self.render
            ).pack(anchor="w", pady=3)
        ttk.Label(display, text="Trail history / samples", style="Muted.TLabel").pack(
            anchor="w", pady=(12, 5)
        )
        history = ttk.Combobox(
            display, textvariable=self.history,
            values=("0", "50", "100", "250", "500", "All"),
            state="readonly", width=16,
        )
        history.pack(anchor="w")
        history.bind("<<ComboboxSelected>>", lambda event: self.render())
        ttk.Button(
            display, text="Reset view", command=self.reset_view
        ).pack(fill="x", pady=(12, 0))

        help_section = self.section(sections, "CONTROLS", expanded=False)
        ttk.Label(
            help_section,
            text=(
                "Space   Play / pause\n"
                "← / →   Step one sample\n\n"
                "Use the plot toolbar to pan,\nzoom, or export an image.\n\n"
                "Trails assume consecutive rows\nbelong to one trajectory."
            ),
            justify="left", style="Muted.TLabel",
        ).pack(anchor="w")

        transport = ttk.Frame(self.root, padding=(18, 12))
        transport.pack(fill="x", padx=18, pady=(10, 0))
        self.play_button = ttk.Button(
            transport, text="Play", command=self.toggle_play, width=8
        )
        self.play_button.pack(side="left")
        ttk.Button(
            transport, text="|←", command=lambda: self.seek(0)
        ).pack(side="left", padx=(8, 3))
        ttk.Button(
            transport, text="←", command=lambda: self.step(-1)
        ).pack(side="left", padx=3)
        ttk.Button(
            transport, text="→", command=lambda: self.step(1)
        ).pack(side="left", padx=3)
        self.slider = ttk.Scale(
            transport, from_=0, to=1, command=self.scrub
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=16)
        ttk.Label(transport, text="samples/s", style="Muted.TLabel").pack(
            side="left", padx=(0, 8)
        )
        ttk.Combobox(
            transport, textvariable=self.speed,
            values=("5", "15", "30", "60"), state="readonly", width=5,
        ).pack(side="left")
        ttk.Checkbutton(
            transport, text="Loop", variable=self.loop
        ).pack(side="left", padx=(14, 0))
        ttk.Label(
            self.root, textvariable=self.status, style="Header.TLabel",
            padding=(24, 10),
        ).pack(fill="x")

        self.style_axis(self.ax, "x / source units", "y / source units")
        self.canvas.draw_idle()

    @staticmethod
    def section(parent, title, expanded=True):
        section = Foldout(parent, title, expanded)
        section.pack(fill="x", padx=8, pady=(8, 0))
        return section.content

    @staticmethod
    def style_axis(axis, xlabel, ylabel):
        axis.set_facecolor(BG)
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.set_xlabel(xlabel, color=MUTED, fontsize=9)
        axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
        axis.grid(True, color=GRID, linewidth=0.6, alpha=0.65)
        axis.set_axisbelow(True)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Open trajectory",
            initialdir=ROOT / "predictions",
            filetypes=[("CSV datasets", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            positions, velocities, headers = load_data(path)
        except (OSError, ValueError, csv.Error, KeyError) as error:
            messagebox.showerror("Could not load trajectory", str(error))
            return

        self.positions = positions
        self.velocities = velocities
        self.frame = 0
        self.playing = False
        self.play_button.configure(text="Play")
        self.source_label.set(Path(path).name)
        self.sample_label.set(
            f"{len(positions):,} samples  /  3 bodies\n"
            f"{len(headers)} source columns"
        )
        self.slider.configure(to=max(1, len(positions) - 1))
        self.ax.clear()
        self.style_axis(self.ax, "x / source units", "y / source units")
        self.ax.set_aspect("equal", adjustable="box")
        self.trails = [
            self.ax.plot([], [], color=color, linewidth=1.25, alpha=0.65)[0]
            for color in COLORS
        ]
        self.markers = [
            self.ax.plot(
                [], [], "o", color=color, markersize=7,
                markeredgecolor=BG, markeredgewidth=1.5,
            )[0]
            for color in COLORS
        ]
        self.labels = [
            self.ax.annotate(
                f" {i + 1:02}", (0, 0), xytext=(8, 8),
                textcoords="offset points", color=color, fontsize=9,
            )
            for i, color in enumerate(COLORS)
        ]
        self.build_diagnostics()
        self.reset_view()

    def build_diagnostics(self):
        for axis in (self.distance_ax, self.speed_ax):
            axis.clear()
        self.style_axis(self.distance_ax, "Sample index", "Separation / source units")
        self.style_axis(self.speed_ax, "Sample index", "Speed / source units")
        samples = np.arange(len(self.positions))
        for (a, b), color in zip(((0, 1), (0, 2), (1, 2)), COLORS):
            distance = np.linalg.norm(
                self.positions[:, a] - self.positions[:, b], axis=1
            )
            self.distance_ax.plot(
                samples, distance, color=color, linewidth=1.2,
                label=f"{a + 1:02} — {b + 1:02}",
            )
        if self.velocities is not None:
            speeds = np.linalg.norm(self.velocities, axis=2)
            for i, color in enumerate(COLORS):
                self.speed_ax.plot(
                    samples, speeds[:, i], color=color,
                    linewidth=1.2, label=BODY_NAMES[i],
                )
        else:
            self.speed_ax.text(
                0.5, 0.5, "Velocity columns are not available",
                transform=self.speed_ax.transAxes,
                ha="center", color=MUTED,
            )
        for axis in (self.distance_ax, self.speed_ax):
            if axis.get_legend_handles_labels()[0]:
                axis.legend(
                    facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
                    fontsize=9, loc="upper right",
                )
        self.cursors = [
            axis.axvline(0, color=TEXT, alpha=0.5, linewidth=0.8)
            for axis in (self.distance_ax, self.speed_ax)
        ]

    def fit_view(self, points):
        low = points.min(axis=(0, 1))
        high = points.max(axis=(0, 1))
        center = (low + high) / 2
        span = max(float(np.max(high - low)), 1.0) * 0.60
        self.ax.set_xlim(center[0] - span, center[0] + span)
        self.ax.set_ylim(center[1] - span, center[1] + span)

    def reset_view(self):
        if self.positions is not None:
            self.fit_view(self.positions)
            self.render()

    def seek(self, frame):
        if self.positions is None:
            return
        self.frame = max(0, min(int(frame), len(self.positions) - 1))
        self.render()

    def scrub(self, value):
        if not self.syncing:
            self.seek(round(float(value)))

    def step(self, amount):
        self.playing = False
        self.play_button.configure(text="Play")
        self.seek(self.frame + amount)

    def toggle_play(self, event=None):
        # Do not consume Space when a form control has keyboard focus.
        if event is not None and isinstance(
            self.root.focus_get(), (ttk.Combobox, ttk.Button, ttk.Checkbutton)
        ):
            return
        if self.positions is not None:
            if not self.playing and self.frame == len(self.positions) - 1:
                self.seek(0)
            self.playing = not self.playing
            self.play_button.configure(text="Pause" if self.playing else "Play")
        return "break"

    def render(self):
        if self.positions is None:
            return

        frame = self.frame
        current = self.positions[frame]
        length = self.history.get()
        start = 0 if length == "All" else max(0, frame - int(length))
        recent = self.positions[start:frame + 1]

        for i in range(3):
            self.trails[i].set_data(recent[:, i, 0], recent[:, i, 1])
            self.markers[i].set_data([current[i, 0]], [current[i, 1]])
            self.labels[i].xy = tuple(current[i])
            self.labels[i].set_visible(self.show_labels.get())
            self.coordinate_labels[i].set(
                f"x {current[i, 0]: .5g}\ny {current[i, 1]: .5g}"
            )
            velocity_values = ("—", "—", "—")
            if self.velocities is not None:
                vx, vy = self.velocities[frame, i]
                speed = np.linalg.norm(self.velocities[frame, i])
                self.velocity_labels[i].set(
                    f"vx {vx: .5g}\nvy {vy: .5g}\n|v| {speed: .5g}"
                )
                velocity_values = tuple(f"{value:.6g}" for value in (vx, vy, speed))
            else:
                self.velocity_labels[i].set("Not supplied by CSV")
            self.table.item(
                str(i),
                values=(
                    BODY_NAMES[i], f"{current[i, 0]:.6g}",
                    f"{current[i, 1]:.6g}", *velocity_values,
                ),
            )

        self.ax.grid(self.show_grid.get(), color=GRID, linewidth=0.6, alpha=0.65)
        if self.follow.get():
            self.fit_view(recent)
        for cursor in self.cursors:
            cursor.set_xdata([frame, frame])

        self.syncing = True
        self.slider.set(frame)
        self.syncing = False
        self.status.set(
            f"{'PLAYING' if self.playing else 'PAUSED'}   /   "
            f"Sample {frame:,} of {len(self.positions) - 1:,}   /   "
            "Equal spatial scale   /   Source units"
        )
        tab = self.tabs.index(self.tabs.select())
        if tab == 0:
            self.canvas.draw_idle()
        elif tab == 1:
            self.diagnostic_canvas.draw_idle()

    def tick(self):
        if self.playing and self.positions is not None:
            if self.frame < len(self.positions) - 1:
                self.frame += 1
            elif self.loop.get():
                self.frame = 0
            else:
                self.playing = False
                self.play_button.configure(text="Play")
            self.render()
        self.timer = self.root.after(
            max(1, round(1000 / float(self.speed.get()))), self.tick
        )

    def close(self):
        self.root.after_cancel(self.timer)
        self.root.destroy()


def main():
    root = tk.Tk()
    GravityViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()