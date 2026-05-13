import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import scipy.interpolate as spi

# ============================================================
# Define colors based on your figures
# ============================================================

color_data   = "#1f77b4"   # data blue
color_maxima = "#ff0000"   # maxima red dots
color_minima = "#0000ff"   # minima blue dots
color_upper  = "#ff7f0e"   # upper envelope orange
color_lower  = "#2ca02c"   # lower envelope green
color_mean   = "#d62728"   # mean envelope red
color_imf    = "#00cfd5"   # IMF purple

LINE_WIDTH = 3.0

# ============================================================
# Generate signal
# ============================================================

np.random.seed(100)
data = np.random.random(100) - 0.5

# ============================================================
# Find local extrema
# ============================================================

max_peaks_tuple = argrelextrema(data, np.greater)
min_peaks_tuple = argrelextrema(data, np.less)

max_peaks = list(max_peaks_tuple[0])
min_peaks = list(min_peaks_tuple[0])

# ============================================================
# Plot local extrema
# ============================================================

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data, color=color_data, linewidth=LINE_WIDTH)

plt.scatter(
    max_peaks_tuple,
    data[max_peaks_tuple],
    c=color_maxima
)

plt.scatter(
    min_peaks_tuple,
    data[min_peaks_tuple],
    c=color_minima
)

plt.gca().set_facecolor("none")
plt.title("Find Local Extrema")

plt.show()

# ============================================================
# Cubic spline interpolation
# ============================================================

index = list(range(len(data)))

# Upper envelope
ipo3_max = spi.splrep(
    max_peaks,
    data[max_peaks],
    k=3
)

iy3_max = spi.splev(index, ipo3_max)

# Lower envelope
ipo3_min = spi.splrep(
    min_peaks,
    data[min_peaks],
    k=3
)

iy3_min = spi.splev(index, ipo3_min)

# Mean envelope
iy3_mean = (iy3_max + iy3_min) / 2

# ============================================================
# Plot cubic spline interpolation
# ============================================================

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data[2:-3], color=color_data, linewidth=LINE_WIDTH)
plt.plot(iy3_max[2:-3], color=color_upper, linewidth=LINE_WIDTH)
plt.plot(iy3_min[2:-3], color=color_lower, linewidth=LINE_WIDTH)
plt.plot(iy3_mean[2:-3], color=color_mean, linewidth=LINE_WIDTH)

plt.gca().set_facecolor("none")
plt.title("Cubic Spline Interpolation")

plt.show()

# ============================================================
# Define IMF1
# ============================================================

imf1 = data - iy3_mean

plt.figure(figsize=(18, 6), facecolor="none")

plt.plot(data[2:-3], color=color_data, linewidth=LINE_WIDTH)
plt.plot(imf1[2:-3], color=color_imf, linewidth=LINE_WIDTH)
plt.plot(iy3_mean[2:-3], color=color_mean, linewidth=LINE_WIDTH)

plt.gca().set_facecolor("none")
plt.title("IMF1")

plt.show()
