import tkinter as tk
from tkinter import messagebox
import random
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import Point


class SyntheticDatasetGenerator:
    def __init__(self, map_size, iterations, update_frequency, variance, random_point_ratio, change_factor, separation_factor, frequency_factor, canvas_size=500):
        """
        Initializes the generator.

        Parameters:
            map_size (int): Size of the square map (map is map_size x map_size).
            iterations (int): Total number of iterations to generate.
            update_frequency (int): Number of iterations between updates to points.
            variance (float): Max variance to apply to points (for small positional changes).
            random_point_ratio (float): Ratio of points generated completely randomly (0-1).
            change_factor (float): Controls how much the dataset changes (0 = no change, 1 = max change).
            separation_factor (float): Minimum distance between points.
            frequency_factor (float): How much frequency matters for regions of interest (0-1).
        """
        self.map_size = map_size
        self.iterations = iterations
        self.update_frequency = update_frequency
        self.variance = variance * change_factor
        self.random_point_ratio = random_point_ratio * change_factor
        self.change_factor = change_factor
        self.separation_factor = separation_factor
        self.frequency_factor = frequency_factor
        self.fixed_points = []
        self.regions_of_interest = []
        self.all_iterations = []
        self.canvas_size = canvas_size
        self.canvas = None
        self.scale_factor = self.canvas_size / self.map_size
        self.offset_x = 0
        self.offset_y = 0
        self.drawn_points = []  # To store points drawn persistently

    def hand_pick_points(self, num_points):
        self.fixed_points = []
        selected_points = []

        def on_canvas_click(event):
            # Map the canvas coordinates to the actual map size
            x = (event.x / self.scale_factor) + self.offset_x
            y = (event.y / self.scale_factor) + self.offset_y
            selected_points.append((x, y))
            self.drawn_points.append((x, y))  # Store persistently
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="red")
            label_status.config(text=f"Points Selected: {len(selected_points)} / {num_points}")

            if len(selected_points) == num_points:
                messagebox.showinfo("Selection Complete", "All points have been selected!")
                root.destroy()

        def zoom(event):
            nonlocal self
            factor = 1.1 if event.delta > 0 else 0.9
            self.scale_factor *= factor

            # Adjust the offset to keep the zoom centered
            mouse_x = event.x / self.scale_factor + self.offset_x
            mouse_y = event.y / self.scale_factor + self.offset_y
            self.offset_x = mouse_x - (event.x / (self.scale_factor * factor))
            self.offset_y = mouse_y - (event.y / (self.scale_factor * factor))

            self.update_canvas()

        def reset_zoom(event):
            self.scale_factor = self.canvas_size / self.map_size
            self.offset_x = 0
            self.offset_y = 0
            self.update_canvas()

        root = tk.Tk()
        root.title("Point Selection")
        root.geometry(f"{self.canvas_size + 100}x{self.canvas_size + 100}")

        label_status = tk.Label(root, text=f"Points Selected: 0 / {num_points}")
        label_status.pack()

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", on_canvas_click)
        self.canvas.bind("<MouseWheel>", zoom)
        self.canvas.bind("<Button-3>", reset_zoom)

        self.update_canvas()

        root.mainloop()

        self.fixed_points = [{'x': x, 'y': y, 'type': 'fixed', 'frequency': random.randint(1, 5)} for x, y in
                             selected_points]

    def hand_draw_regions(self):
        self.regions_of_interest = []
        current_polygon = []
        all_polygons = []

        def on_canvas_click(event):
            x = (event.x / self.scale_factor) + self.offset_x
            y = (event.y / self.scale_factor) + self.offset_y
            current_polygon.append((x, y))
            self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="blue")

        def finish_polygon():
            if len(current_polygon) >= 3:
                poly = Polygon(current_polygon)
                all_polygons.append(poly)

                for i in range(len(current_polygon)):
                    x1, y1 = current_polygon[i]
                    x2, y2 = current_polygon[(i + 1) % len(current_polygon)]
                    self.canvas.create_line(
                        (x1 - self.offset_x) * self.scale_factor,
                        (y1 - self.offset_y) * self.scale_factor,
                        (x2 - self.offset_x) * self.scale_factor,
                        (y2 - self.offset_y) * self.scale_factor,
                        fill="green"
                    )
                current_polygon.clear()
            else:
                messagebox.showerror("Error", "A polygon must have at least 3 points!")

        def finalize_regions():
            if all_polygons:
                self.regions_of_interest = all_polygons
                messagebox.showinfo("Selection Complete", "Regions have been finalized!")
                root.destroy()
            else:
                messagebox.showerror("Error", "You must draw at least one region!")

        def zoom(event):
            nonlocal self
            factor = 1.1 if event.delta > 0 else 0.9
            self.scale_factor *= factor

            # Adjust the offset to keep the zoom centered
            mouse_x = event.x / self.scale_factor + self.offset_x
            mouse_y = event.y / self.scale_factor + self.offset_y
            self.offset_x = mouse_x - (event.x / (self.scale_factor * factor))
            self.offset_y = mouse_y - (event.y / (self.scale_factor * factor))

            self.update_canvas()

        def reset_zoom(event):
            self.scale_factor = self.canvas_size / self.map_size
            self.offset_x = 0
            self.offset_y = 0
            self.update_canvas()

        root = tk.Tk()
        root.title("Region Selection")
        root.geometry(f"{self.canvas_size + 100}x{self.canvas_size + 100}")

        button_finish = tk.Button(root, text="Finish Polygon", command=finish_polygon)
        button_finish.pack()

        button_finalize = tk.Button(root, text="Finalize Regions", command=finalize_regions)
        button_finalize.pack()

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", on_canvas_click)
        self.canvas.bind("<MouseWheel>", zoom)
        self.canvas.bind("<Button-3>", reset_zoom)

        self.update_canvas()

        root.mainloop()

    def update_canvas(self):
        if self.canvas is not None:
            self.canvas.delete("all")
            self.draw_grid()
            self.redraw_points()

    def draw_grid(self):
        # Calculate the visible area based on the current scale and offset
        start_x = self.offset_x
        start_y = self.offset_y
        end_x = self.offset_x + self.canvas_size / self.scale_factor
        end_y = self.offset_y + self.canvas_size / self.scale_factor

        # Determine grid step size for axis labels
        grid_step = 10
        while (end_x - start_x) / grid_step > 20:
            grid_step *= 2

        # Draw vertical grid lines and labels
        x = start_x - (start_x % grid_step)
        while x <= end_x:
            screen_x = (x - self.offset_x) * self.scale_factor
            self.canvas.create_line(screen_x, 0, screen_x, self.canvas_size, fill="gray")
            self.canvas.create_text(screen_x, self.canvas_size - 10, text=f"{round(x, 2)}", fill="black",
                                    font=("Arial", 8))
            x += grid_step

        # Draw horizontal grid lines and labels
        y = start_y - (start_y % grid_step)
        while y <= end_y:
            screen_y = (y - self.offset_y) * self.scale_factor
            self.canvas.create_line(0, screen_y, self.canvas_size, screen_y, fill="gray")
            self.canvas.create_text(10, screen_y, text=f"{round(y, 2)}", fill="black", font=("Arial", 8), anchor="w")
            y += grid_step

    def redraw_points(self):
        for x, y in self.drawn_points:
            screen_x = (x - self.offset_x) * self.scale_factor
            screen_y = (y - self.offset_y) * self.scale_factor
            self.canvas.create_oval(screen_x - 5, screen_y - 5, screen_x + 5, screen_y + 5, fill="red")
    def simulate_iterations(self):
        for iteration in range(self.iterations):
            current_points = []

            for point in self.fixed_points:
                x, y = point['x'], point['y']

                if random.random() < self.change_factor:
                    x += random.uniform(-self.variance, self.variance)
                    y += random.uniform(-self.variance, self.variance)

                x = max(0, min(self.map_size, x))
                y = max(0, min(self.map_size, y))

                current_points.append({'x': x, 'y': y, 'type': 'fixed', 'frequency': point['frequency']})

            for region in self.regions_of_interest:
                if random.random() < self.frequency_factor:
                    num_points_in_region = max(1, int(len(self.fixed_points) * self.frequency_factor))
                    for _ in range(num_points_in_region):
                        while True:
                            x = random.uniform(0, self.map_size)
                            y = random.uniform(0, self.map_size)
                            if region.contains(Point(x, y)):
                                current_points.append({'x': x, 'y': y, 'type': 'region', 'frequency': 1})
                                break

            num_random_points = int(len(self.fixed_points) * self.random_point_ratio)
            for _ in range(num_random_points):
                x = random.uniform(0, self.map_size)
                y = random.uniform(0, self.map_size)

                if self.separation_factor > 0:
                    if any((abs(x - p['x']) < self.separation_factor and abs(y - p['y']) < self.separation_factor)
                           for p in current_points):
                        continue

                current_points.append({'x': x, 'y': y, 'type': 'random', 'frequency': 1})

            for point in current_points:
                point['iteration'] = iteration
            self.all_iterations.extend(current_points)

    def save_iterations(self, output_file):
        if not self.all_iterations:
            print("No iterations to save. Run 'simulate_iterations' first.")
            return

        df = pd.DataFrame(self.all_iterations)
        df.to_csv(output_file, index=False)
        print(f"Iterations saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    map_size = 100
    generator = SyntheticDatasetGenerator(map_size, 50, 5, 5.0, 0.3, 0.7, 10.0, 0.8)
    generator.hand_pick_points(300)
    generator.hand_draw_regions()
    generator.simulate_iterations()
    generator.save_iterations("../Data/hand_picked_points.csv")

