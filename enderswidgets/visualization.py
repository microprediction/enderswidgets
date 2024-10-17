import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from IPython.display import display, clear_output
from typing import Deque, Optional, List, Tuple

from enderswidgets.streams import Prediction, StreamPoint

import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", message=".*Attempting to set identical low and high xlims.*")

# Define margin adjustment functions
def margin_down(y: float, margin: float = 0.0005) -> float:
    return y * (1 + margin) if y < 0 else y * (1 - margin)

def margin_up(y: float, margin: float = 0.0005) -> float:
    return y * (1 - margin) if y < 0 else y * (1 + margin)

# Define optimized TimeSeriesVisualizer class
class TimeSeriesVisualizer:
    def __init__(self, max_points: Optional[int] = None, update_interval: int = 5):
        self.max_points = max_points
        self.update_interval = update_interval

        # Initialize deques to store time and value data
        self.times: Deque[int] = deque(maxlen=max_points)
        self.values: Deque[float] = deque(maxlen=max_points)

        # Initialize event indicators
        self.event_lines = []
        self.event_shades = []

        # Counter for updates
        self.update_counter = 0

        # Initialize figure and axis
        self.fig = None
        self.ax = None
        self.line = None
        self.display_handle = None

        # Create the initial plot
        self.create_plot()

    def create_plot(self):
        """Create a new figure and axis for plotting."""
        if self.fig is not None:
            plt.close(self.fig)  # Close the existing figure
        self.fig, self.ax = plt.subplots(figsize=(20, 10))  # Increase figure size
        self._setup_plot_style()
        self.line, = self.ax.plot([], [], color='lightgrey', label='Realized')
        self.ax.legend()

    def _setup_plot_style(self):
        """Set up the plot style."""
        plt.style.use('dark_background')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('Value', color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

    def process(self, data: StreamPoint, pred: Prediction):
        """Update the plot with new data from StreamPoint."""
        self.times.append(data.ndx)
        self.values.append(data.value)
        if pred.value != 0:
            self.show_decision(data.ndx, pred.value)
        self.update_counter += 1

    def show_decision(self, n: int, decision: float):
        """Show a decision on the plot."""
        color = 'green' if decision > 0 else 'red'
        self.add_event_shade(n, n + 10, color=color, alpha=0.3)

    def add_event_shade(self, start_time: int, end_time: int, color: str = 'yellow', alpha: float = 0.3, label: Optional[str] = None):
        """Add a shaded region to indicate an event period."""
        shade = self.ax.axvspan(start_time, end_time, color=color, alpha=alpha, label=label)
        self.event_shades.append(shade)
        if label:
            self.ax.legend()
        self.display()  # Update display to show the new shade

    def display(self):
        """Update the plot with current data."""
        if len(self.times) <= 1 or len(self.values) <= 1:
            return
        # Convert deques to numpy arrays for efficient plotting
        times_array = np.array(self.times)
        values_array = np.array(self.values)

        # Update realized values line
        self.line.set_data(times_array, values_array)

        # Adjust axis limits
        if len(times_array) > 0 and len(values_array) > 0:
            self.ax.set_xlim(margin_down(np.min(times_array)), margin_up(np.max(times_array)))
            self.ax.set_ylim(margin_down(np.min(values_array)), margin_up(np.max(values_array)))

        # Ensure all elements are visible
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Update display in Jupyter
        if self.display_handle is None:
            self.display_handle = display(self.fig, display_id=True)
        else:
            self.display_handle.update(self.fig)


    def clear(self):
        """Clear the plot and reset data."""
        self.times.clear()
        self.values.clear()
        self.line.set_data([], [])

        # Remove event indicators
        for line in self.event_lines:
            line.remove()
        self.event_lines.clear()
        for shade in self.event_shades:
            shade.remove()
        self.event_shades.clear()

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.display()  # Update display after clearing

    def close(self):
        """Close the figure to free up memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.line = None
            self.display_handle = None
