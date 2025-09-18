import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.stats as stats

NUM_BINS = 100
N = 10000
WINDOW_WIDTH, WINDOW_HEIGHT = 1400, 700

# Interactive histogram canvas - allows dragging bars to change heights
class HistogramCanvas(FigureCanvas):
    def __init__(self, data, bins, on_update):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data = data
        self.bins = bins
        self.on_update = on_update
        self.bar_patches = None
        self._dragging = False
        self.plot_histogram()
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("button_release_event", self.on_release)

    def plot_histogram(self):
        self.ax.clear()
        counts, bins, patches = self.ax.hist(self.data, bins=self.bins, edgecolor='black', picker=True)
        self.bar_patches = patches
        self.ax.set_title("Histogram")
        self.fig.tight_layout()
        self.draw()

    def set_bar_height_at_x(self, x, y):
        for i, patch in enumerate(self.bar_patches):
            left = patch.get_x()
            right = left + patch.get_width()
            if left <= x < right:
                new_height = max(y, 0)
                patch.set_height(new_height)

                lefts = [p.get_x() for p in self.bar_patches]
                widths = [p.get_width() for p in self.bar_patches]
                heights = [p.get_height() for p in self.bar_patches]
                new_data = []
                for l, w, h in zip(lefts, widths, heights):
                    new_data += [l + w/2] * int(round(h))
                self.data = np.array(new_data)
                self.on_update(self.data)
                self.plot_histogram()
                break

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self._dragging = True
        self.set_bar_height_at_x(event.xdata, event.ydata)

    def on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.set_bar_height_at_x(event.xdata, event.ydata)

    def on_release(self, event):
        self._dragging = False

#QQ plot canvas - updates when histogram changes
class QQPlotCanvas(FigureCanvas):
    def __init__(self, data):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data = data
        self.plot_qq()

    def plot_qq(self):
        self.ax.clear()
        if len(self.data) > 0:
            stats.probplot(self.data, dist="norm", plot=self.ax)
        self.ax.set_title("QQ Plot")
        self.ax.set_ylabel("Sample Quantiles") 
        self.ax.set_xlabel("Theoretical Quantiles") 
        self.draw()

    def update_data(self, data):
        self.data = data
        self.plot_qq()

# Main app window - handles updating qq plot when histogram changes
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Histogram and QQ Plot")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.bins = NUM_BINS

        # Use percent point function (inverse CDF) for a perfect normal
        self.data = stats.norm.ppf(np.linspace(1/(N+1), N/(N+1), N))

        central = QWidget()
        layout = QHBoxLayout(central)
        self.hist_canvas = HistogramCanvas(self.data, self.bins, self.update_qq)
        self.qq_canvas = QQPlotCanvas(self.data)
        layout.addWidget(self.hist_canvas)
        layout.addWidget(self.qq_canvas)
        self.setCentralWidget(central)

    def update_qq(self, new_data):
        self.qq_canvas.update_data(new_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())