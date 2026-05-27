"""
Interactive canvas with visual cell deformation (stickiness appearance).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from typing import Optional, Dict, List
import os

from core.tissue import Tissue
from core.cell import Cell
from parser.gro_parser import load_gro_file

from PIL import Image, ImageTk  # Add Pillow as dependency


class GrowthCanvas:
    """Interactive canvas for the growth simulation."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cell Growth Simulator")
        self.root.geometry("1400x900")
        
        self.tissue: Optional[Tissue] = None
        self.running: bool = False
        self.simulation_speed: int = 200
        
        self.view_center = np.array([0.0, 0.0])
        self.view_scale = 80.0
        self.selected_cell_id: Optional[str] = None
        self.loaded_filepath: Optional[str] = None
        
        self.drag_start = None
        self.drag_start_center = None
        
        self._setup_ui()
        self._bind_events()

        # Add these for caching
        self._gradient_cache: Optional[ImageTk.PhotoImage] = None
        self._gradient_cache_key: Optional[tuple] = None  # (gradient_name, view_center, view_scale, canvas_size)
        self._cell_items: Dict[str, List[int]] = {}  # cell_id -> canvas item IDs
    
    def _setup_ui(self):
        """Create UI layout."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_panel = ttk.Frame(self.main_frame, width=280)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_panel.pack_propagate(False)
        
        # File section
        file_frame = ttk.LabelFrame(self.control_panel, text="File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load .gro File",
                   command=self._load_file).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Reset",
                   command=self._reset).pack(fill=tk.X, padx=5, pady=2)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", wraplength=250)
        self.file_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Simulation section
        sim_frame = ttk.LabelFrame(self.control_panel, text="Simulation")
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(sim_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.step_btn = ttk.Button(btn_frame, text="Step",
                                   command=self._step, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.step10_btn = ttk.Button(btn_frame, text="Step 10",
                                     command=lambda: self._step_n(10), state=tk.DISABLED)
        self.step10_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.run_btn = ttk.Button(sim_frame, text="Run",
                                  command=self._toggle_run, state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(sim_frame, text="Speed (ms/step):").pack(padx=5, pady=(5,0))
        self.speed_var = tk.IntVar(value=200)
        speed_frame = ttk.Frame(sim_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.speed_scale = ttk.Scale(speed_frame, from_=20, to=1000,
                                     variable=self.speed_var, orient=tk.HORIZONTAL)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.speed_label = ttk.Label(speed_frame, text="200", width=5)
        self.speed_label.pack(side=tk.RIGHT)
        self.speed_var.trace_add('write', self._update_speed_label)
        
        # View section
        view_frame = ttk.LabelFrame(self.control_panel, text="View")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_gradients_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Show Gradient Background",
                        variable=self.show_gradients_var,
                        command=self._redraw).pack(anchor=tk.W, padx=5)
        
        self.show_centers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Show Cell Centers",
                        variable=self.show_centers_var,
                        command=self._redraw).pack(anchor=tk.W, padx=5)

        self.auto_fit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Auto-fit during Run",
                        variable=self.auto_fit_var).pack(anchor=tk.W, padx=5)
        
        ttk.Label(view_frame, text="Gradient to display:").pack(padx=5, pady=(5,0))
        self.gradient_var = tk.StringVar(value="None")
        self.gradient_combo = ttk.Combobox(view_frame, textvariable=self.gradient_var,
                                           state="readonly", values=["None"])
        self.gradient_combo.pack(fill=tk.X, padx=5, pady=2)
        self.gradient_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        
        ttk.Button(view_frame, text="Fit View",
                command=self._fit_view).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(view_frame, text="Reset View",
                   command=self._reset_view).pack(fill=tk.X, padx=5, pady=2)

        zoom_frame = ttk.Frame(view_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(zoom_frame, text="Zoom +",
                   command=lambda: self._zoom(1.3)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(zoom_frame, text="Zoom −",
                   command=lambda: self._zoom(1/1.3)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(self.control_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30,
                                  state=tk.DISABLED, font=('Courier', 9))
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Cell info section
        info_frame = ttk.LabelFrame(self.control_panel, text="Selected Cell")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cell_info_text = tk.Text(info_frame, height=12, width=30,
                                      state=tk.DISABLED, font=('Courier', 9))
        self.cell_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _bind_events(self):
        """Bind input events."""
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<Configure>", self._on_resize)
        
        self.root.bind("<space>", lambda e: self._step() if self.tissue else None)
        self.root.bind("<Return>", lambda e: self._toggle_run() if self.tissue else None)
        self.root.bind("f", lambda e: self._fit_view())
        self.root.bind("r", lambda e: self._reset_view())
    
    def _update_speed_label(self, *args):
        self.speed_label.config(text=str(self.speed_var.get()))
    
    def _load_file(self):
        """Load a .gro file."""
        filepath = filedialog.askopenfilename(
            title="Select .gro file",
            filetypes=[("Growth files", "*.gro"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.tissue = Tissue()
            load_gro_file(filepath, self.tissue)
            self.loaded_filepath = filepath            

            # Update gradient combo
            gradient_names = ["None"]
            
            # Add static gradients
            for name, g in self.tissue.gradient_field.gradients.items():
                if g.fixed_position is None and g.attached_to_cell_id is None:
                    gradient_names.append(name)
            
            # Add pattern options for any emitted micro-gradient prefixes
            seen_prefixes = set()
            for rule_list in self.tissue.division_rules.values():
                for rule in rule_list:
                    if rule.generated_gradient:
                        seen_prefixes.add(rule.generated_gradient.name_prefix + "*")
            for prefix in sorted(seen_prefixes):
                gradient_names.append(prefix)
            
            self.gradient_combo["values"] = gradient_names
            if len(gradient_names) > 1:
                self.gradient_var.set(gradient_names[1])
            
            self.step_btn.config(state=tk.NORMAL)
            self.step10_btn.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)
            
            self.file_label.config(text=os.path.basename(filepath))
            
            self._update_stats()
            self.status_var.set(f"Loaded: {filepath}")
            # Delay fit_view to ensure canvas is sized
            self.root.after(50, self._fit_view)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error loading file")
    
    def _reset(self):
        """Reset simulation. If a file is loaded, restart it."""
        self.running = False
        self.run_btn.config(text="Run")
        self.selected_cell_id = None

        if self.loaded_filepath:
            try:
                self.tissue = Tissue()
                load_gro_file(self.loaded_filepath, self.tissue)
                # refresh gradient combo
                gradient_names = ["None"]
                for name, g in self.tissue.gradient_field.gradients.items():
                    if g.fixed_position is None and g.attached_to_cell_id is None:
                        gradient_names.append(name)
                seen_prefixes = set()
                for rule_list in self.tissue.division_rules.values():
                    for rule in rule_list:
                        if rule.generated_gradient:
                            seen_prefixes.add(rule.generated_gradient.name_prefix + "*")
                for prefix in sorted(seen_prefixes):
                    gradient_names.append(prefix)
                self.gradient_combo["values"] = gradient_names

                self._fit_view()
                self._update_stats()
                self._update_cell_info()
                self.status_var.set(f"Reset: {os.path.basename(self.loaded_filepath)}")
                return
            except Exception:
                pass

        self.tissue = None
        self.loaded_filepath = None
        self.step_btn.config(state=tk.DISABLED)
        self.step10_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)
        self.file_label.config(text="No file loaded")
        self.gradient_combo["values"] = ["None"]
        self.gradient_var.set("None")
        self._redraw()
        self._update_stats()
        self._update_cell_info()
        self.status_var.set("Reset")
    
    def _step(self):
        """Single simulation step."""
        if self.tissue:
            divisions, deaths, differentiations = self.tissue.step()
            self._redraw()
            self._update_stats()
            if self.selected_cell_id:
                self._update_cell_info()
            self.status_var.set(
                f"Step {self.tissue.simulation_step}: "
                f"{len(divisions)} div, {len(deaths)} deaths, {len(differentiations)} diff"
            )
    
    def _step_n(self, n: int):
        """Multiple simulation steps without intermediate redraws."""
        if self.tissue:
            total_div, total_deaths, total_diff = self.tissue.run_steps(n)
            self._redraw()
            self._update_stats()
            if self.selected_cell_id:
                self._update_cell_info()
            self.status_var.set(
                f"Step {self.tissue.simulation_step}: "
                f"{total_div} div, {total_deaths} deaths, {total_diff} diff in {n} steps"
            )
    
    def _toggle_run(self):
        """Toggle continuous running."""
        self.running = not self.running
        self.run_btn.config(text="Stop" if self.running else "Run")
        if self.running:
            self._run_loop()
    
    def _run_loop(self):
        """Continuous simulation loop with adaptive batching."""
        if self.running and self.tissue:
            self.simulation_speed = self.speed_var.get()
            
            # Batch more steps when running fast
            steps_per_frame = 1
            if self.simulation_speed < 50:
                steps_per_frame = 3
            elif self.simulation_speed < 100:
                steps_per_frame = 2
            
            for _ in range(steps_per_frame):
                self.tissue.step()
            
            self._redraw()
            self._update_stats()
            
            # Auto-fit if enabled
            if self.auto_fit_var.get():
                self._fit_view()
            
            self.status_var.set(f"Running - Step {self.tissue.simulation_step}, {len(self.tissue.cells)} cells")
            self.root.after(self.simulation_speed, self._run_loop)
    
    def _fit_view(self):
        """Fit view to all cells."""
        if not self.tissue or not self.tissue.cells:
            self._reset_view()
            return
        
        bounds = self.tissue.get_bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Ensure canvas has been rendered
        self.canvas.update_idletasks()
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Fallback if canvas not yet sized
        if canvas_width < 10:
            canvas_width = 800
        if canvas_height < 10:
            canvas_height = 600
        
        scale_x = canvas_width / width if width > 0 else 80
        scale_y = canvas_height / height if height > 0 else 80
        
        self.view_scale = min(scale_x, scale_y) * 0.85
        self.view_center = np.array([
            (bounds[0] + bounds[2]) / 2,
            (bounds[1] + bounds[3]) / 2
        ])
        
        # Invalidate gradient cache since view changed
        self._gradient_cache_key = None
        
        self._redraw()
    
    def _reset_view(self):
        """Reset view to default."""
        self.view_center = np.array([0.0, 0.0])
        self.view_scale = 80.0
        self._redraw()

    def _zoom(self, factor: float):
        """Zoom by factor, centred on current view."""
        self.view_scale *= factor
        self.view_scale = max(5, min(800, self.view_scale))
        self._redraw()
    
    def _world_to_screen(self, pos: np.ndarray) -> tuple:
        """Convert world to screen coordinates."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        x = canvas_width / 2 + (pos[0] - self.view_center[0]) * self.view_scale
        y = canvas_height / 2 - (pos[1] - self.view_center[1]) * self.view_scale
        
        return (x, y)
    
    def _screen_to_world(self, screen_pos: tuple) -> np.ndarray:
        """Convert screen to world coordinates."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        x = self.view_center[0] + (screen_pos[0] - canvas_width / 2) / self.view_scale
        y = self.view_center[1] - (screen_pos[1] - canvas_height / 2) / self.view_scale
        
        return np.array([x, y])
    
    def _redraw(self):
        """Redraw everything with frustum culling."""
        self.canvas.delete("all")
        
        # Origin marker
        origin = self._world_to_screen(np.array([0.0, 0.0]))
        self.canvas.create_line(origin[0]-10, origin[1], origin[0]+10, origin[1],
                            fill="#333355", width=1)
        self.canvas.create_line(origin[0], origin[1]-10, origin[0], origin[1]+10,
                            fill="#333355", width=1)
        
        if not self.tissue:
            return
        
        # Gradient background
        if self.show_gradients_var.get() and self.gradient_var.get() != "None":
            self._draw_gradient_background()
        
        # Compute view bounds in world coordinates for culling
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        view_min = self._screen_to_world((0, canvas_height))
        view_max = self._screen_to_world((canvas_width, 0))
        margin = 2.0  # World units margin for cells partially visible
        
        # Draw only visible cells
        for cell in self.tissue.cells.values():
            # Frustum culling - skip cells outside view
            if (cell.center[0] + cell.visual_radius < view_min[0] - margin or
                cell.center[0] - cell.visual_radius > view_max[0] + margin or
                cell.center[1] + cell.visual_radius < view_min[1] - margin or
                cell.center[1] - cell.visual_radius > view_max[1] + margin):
                continue
            
            self._draw_cell(cell)
        
        # Selection
        if self.selected_cell_id and self.selected_cell_id in self.tissue.cells:
            self._draw_selection(self.tissue.cells[self.selected_cell_id])
    
    def _draw_gradient_background(self):
        """Draw gradient as cached image - only recompute when view changes."""
        gradient_name = self.gradient_var.get()
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        # Cache key to detect when we need to recompute
        cache_key = (
            gradient_name,
            round(self.view_center[0], 2),
            round(self.view_center[1], 2),
            round(self.view_scale, 1),
            canvas_width,
            canvas_height
        )
        
        if self._gradient_cache_key == cache_key and self._gradient_cache is not None:
            # Use cached image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._gradient_cache, tags="gradient_bg")
            return
        
        # Recompute gradient image (at lower resolution for speed)
        resolution = 4  # Compute every 4th pixel, then scale up
        small_width = canvas_width // resolution
        small_height = canvas_height // resolution
        
        img = Image.new('RGB', (small_width, small_height))
        pixels = img.load()
        
        for py in range(small_height):
            for px in range(small_width):
                screen_x = px * resolution + resolution // 2
                screen_y = py * resolution + resolution // 2
                world_pos = self._screen_to_world((screen_x, screen_y))
                
                if gradient_name.endswith('*'):
                    prefix = gradient_name[:-1]
                    value = self.tissue.gradient_field.get_summed_value(prefix, world_pos)
                else:
                    gradient = self.tissue.gradient_field.get_gradient(gradient_name)
                    value = gradient.get_value(world_pos) if gradient else 0.0
                
                r = int(200 * value)
                g = int(50 * (1 - abs(value - 0.5) * 2))
                b = int(200 * (1 - value))
                pixels[px, py] = (r, g, b)
        
        # Scale up to full resolution
        img = img.resize((canvas_width, canvas_height), Image.NEAREST)
        
        self._gradient_cache = ImageTk.PhotoImage(img)
        self._gradient_cache_key = cache_key
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._gradient_cache, tags="gradient_bg")
    
    def _compute_cell_vertices(self, cell: Cell, num_points: int = 36) -> List[tuple]:
        """
        Compute cell boundary with visual deformation from neighbors.
        
        Creates the "sticky" appearance by flattening the cell boundary
        where neighboring cells are close.
        """
        theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        vertices = []
        
        # Get touching neighbors for deformation
        neighbors = self.tissue.get_touching_neighbors(cell)
        
        for t in theta_values:
            angle_deg = np.degrees(t)
            r = cell.radius
            
            # Flatten toward each neighbor
            for other in neighbors:
                delta = other.center - cell.center
                dist = np.linalg.norm(delta)
                
                if dist < 0.001:
                    continue
                
                # Direction to neighbor
                neighbor_angle = np.degrees(np.arctan2(delta[1], delta[0])) % 360
                
                # Angular distance
                angle_diff = abs(angle_deg - neighbor_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Flatten if pointing toward neighbor
                flatten_range = 45  # degrees
                if angle_diff < flatten_range:
                    # Proximity factor (closer = more flattening)
                    sum_radii = cell.radius + other.radius
                    proximity = 1 - (dist / (sum_radii * 1.2))
                    proximity = max(0, min(1, proximity))
                    
                    # Angular factor (center of contact = most flattening)
                    angle_factor = 1 - (angle_diff / flatten_range)
                    
                    flatten_amount = 0.18 * proximity * angle_factor
                    r = min(r, cell.radius * (1 - flatten_amount))
            
            x = cell.center[0] + r * np.cos(t)
            y = cell.center[1] + r * np.sin(t)
            vertices.append(self._world_to_screen(np.array([x, y])))
        
        return vertices
    
    # REPLACE the _draw_cell method:

    def _draw_cell(self, cell: Cell):
        """Draw a cell - simplified for performance."""
        color = self.tissue.get_cell_color(cell.cell_type)
        center = self._world_to_screen(cell.center)
        
        # Only compute expensive jelly layer if zoomed in enough
        if self.view_scale > 40:
            # Draw simplified jelly layer (just a larger circle, no stipple)
            jelly_radius = cell.visual_radius * self.view_scale
            jelly_color = self._lighten_color(color, 0.4)
            self.canvas.create_oval(
                center[0] - jelly_radius, center[1] - jelly_radius,
                center[0] + jelly_radius, center[1] + jelly_radius,
                fill=jelly_color, outline=""
            )
        
        # Core circle
        radius = cell.physical_radius * self.view_scale
        self.canvas.create_oval(
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
            fill=color, outline="#ffffff", width=1
        )
        
        # Center marker
        if self.show_centers_var.get() or cell.is_anchor:
            size = 4 if cell.is_anchor else 2
            fill = "#ffff00" if cell.is_anchor else "#ffffff"
            self.canvas.create_oval(
                center[0] - size, center[1] - size,
                center[0] + size, center[1] + size,
                fill=fill, outline=""
            )

    def _get_visual_neighbors(self, cell: Cell) -> List[Cell]:
        """Get cells within visual connection range."""
        neighbors = []
        visual_threshold = cell.visual_radius * 2.2
        
        nearby_ids = self.tissue.physics.spatial_grid.get_nearby_cell_ids(
            cell.center, radius=visual_threshold
        )
        
        for other_id in nearby_ids:
            if other_id == cell.id:
                continue
            other = self.tissue.get_cell(other_id)
            if other:
                dist = np.linalg.norm(other.center - cell.center)
                if dist < cell.visual_radius + other.visual_radius:
                    neighbors.append(other)
        
        return neighbors

    def _draw_jelly_layer(self, cell: Cell, neighbors: List[Cell], color: str):
        """Draw the extended 'jelly' membrane that visually connects to neighbors."""
        num_points = 48
        theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        vertices = []
        
        for t in theta_values:
            angle_deg = np.degrees(t)
            r = cell.visual_radius  # Start with extended radius
            
            # Extend further toward neighbors (creates bridges)
            for other in neighbors:
                delta = other.center - cell.center
                dist = np.linalg.norm(delta)
                
                if dist < 0.001:
                    continue
                
                neighbor_angle = np.degrees(np.arctan2(delta[1], delta[0])) % 360
                angle_diff = abs(angle_deg - neighbor_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Within 40 degrees of neighbor direction, extend toward them
                if angle_diff < 40:
                    # How much do visual radii overlap?
                    overlap = (cell.visual_radius + other.visual_radius) - dist
                    if overlap > 0:
                        # Extend radius to bridge the gap
                        angle_factor = 1 - (angle_diff / 40)
                        extension = min(overlap * 0.5, cell.radius * 0.3) * angle_factor
                        r = max(r, cell.visual_radius + extension * 0.5)
            
            x = cell.center[0] + r * np.cos(t)
            y = cell.center[1] + r * np.sin(t)
            vertices.append(self._world_to_screen(np.array([x, y])))
        
        # Draw with transparency effect (stipple pattern simulates alpha)
        flat_vertices = [coord for v in vertices for coord in v]
        if len(flat_vertices) >= 6:
            # Lighter version of color for jelly
            jelly_color = self._lighten_color(color, 0.3)
            self.canvas.create_polygon(
                flat_vertices,
                fill=jelly_color, outline="",
                stipple="gray50",  # Creates transparency effect
                tags=f"jelly_{cell.id}"
            )

    def _draw_core(self, cell: Cell, color: str):
        """Draw the solid core of the cell."""
        center = self._world_to_screen(cell.center)
        radius = cell.physical_radius * self.view_scale
        
        self.canvas.create_oval(
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
            fill=color, outline="#ffffff", width=1,
            tags=f"cell_{cell.id}"
        )

    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by factor (0-1)."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _draw_selection(self, cell: Cell):
        """Draw selection highlight."""
        center = self._world_to_screen(cell.center)
        radius = (cell.radius + 0.15) * self.view_scale
        
        self.canvas.create_oval(
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
            outline="#ffff00", width=3
        )
    
    def _on_click(self, event):
        """Handle click."""
        self.drag_start = (event.x, event.y)
        self.drag_start_center = self.view_center.copy()
        
        if self.tissue:
            world_pos = self._screen_to_world((event.x, event.y))
            cell = self.tissue.get_cell_at_position(world_pos)
            
            self.selected_cell_id = cell.id if cell else None
            self._update_cell_info()
            self._redraw()
    
    def _on_drag(self, event):
        """Handle drag for panning."""
        if self.drag_start and self.drag_start_center is not None:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            
            self.view_center = self.drag_start_center.copy()
            self.view_center[0] -= dx / self.view_scale
            self.view_center[1] += dy / self.view_scale
            
            self._redraw()
    
    def _on_release(self, event):
        """Handle release."""
        self.drag_start = None
        self.drag_start_center = None
    
    def _on_scroll(self, event):
        """Handle scroll for zooming."""
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15
        else:
            factor = 1 / 1.15
        
        mouse_world = self._screen_to_world((event.x, event.y))
        
        self.view_scale *= factor
        self.view_scale = max(10, min(500, self.view_scale))
        
        new_mouse_world = self._screen_to_world((event.x, event.y))
        self.view_center += mouse_world - new_mouse_world
        
        self._redraw()
    
    def _on_resize(self, event):
        """Handle resize."""
        self._redraw()
    
    def _update_stats(self):
        """Update statistics display."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        if self.tissue:
            stats = self.tissue.get_statistics()
            text = f"Step:     {stats['simulation_step']}\n"
            text += f"Cells:    {stats['total_cells']}\n"
            text += f"\nBy type:\n"
            for cell_type, count in sorted(stats['type_counts'].items()):
                text += f"  {cell_type}: {count}\n"
        else:
            text = "No simulation loaded"
        
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state=tk.DISABLED)
    
    def _update_cell_info(self):
        """Update cell info display."""
        self.cell_info_text.config(state=tk.NORMAL)
        self.cell_info_text.delete(1.0, tk.END)
        
        if self.selected_cell_id and self.tissue:
            cell = self.tissue.get_cell(self.selected_cell_id)
            if cell:
                text = f"ID:       {cell.id}\n"
                text += f"Type:     {cell.cell_type}\n"
                text += f"Position: ({cell.center[0]:.3f}, {cell.center[1]:.3f})\n"
                text += f"Radius:   {cell.radius:.3f}\n"
                text += f"Age:      {cell.age}\n"
                text += f"Anchor:   {cell.is_anchor}\n"
                text += f"Divisions:{cell.division_count}\n"
                
                if cell.parent_id:
                    text += f"Parent:   {cell.parent_id}\n"
                if cell.sibling_id:
                    text += f"Sibling:  {cell.sibling_id}\n"
                if cell.generated_gradient_name:
                    text += f"Gradient: {cell.generated_gradient_name}\n"
                
                # Neighbors
                neighbors = self.tissue.get_neighbors(cell)
                text += f"\nNeighbors: {len(neighbors)}\n"
                
                # Gradient values (static gradients only)
                static_gradients = [
                    (name, g) for name, g in self.tissue.gradient_field.gradients.items()
                    if g.attached_to_cell_id is None
                ]
                if static_gradients:
                    text += f"\nGradients:\n"
                    for name, grad in static_gradients:
                        val = grad.get_value(cell.center)
                        text += f"  {name}: {val:.3f}\n"
                
                self.cell_info_text.insert(tk.END, text)
        else:
            self.cell_info_text.insert(tk.END, "Click a cell to select")
        
        self.cell_info_text.config(state=tk.DISABLED)

    def _auto_fit_if_needed(self):
        """Auto-fit view if cells are approaching edge."""
        if not self.tissue or not self.tissue.cells:
            return
        
        bounds = self.tissue.get_bounds()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get current view bounds in world coordinates
        view_min = self._screen_to_world((0, canvas_height))
        view_max = self._screen_to_world((canvas_width, 0))
        
        margin = 1.0  # World units
        
        # Check if tissue bounds exceed view bounds
        if (bounds[0] < view_min[0] + margin or 
            bounds[1] < view_min[1] + margin or
            bounds[2] > view_max[0] - margin or 
            bounds[3] > view_max[1] - margin):
            self._fit_view()

def run_app():
    """Run the application."""
    root = tk.Tk()
    app = GrowthCanvas(root)
    root.mainloop()