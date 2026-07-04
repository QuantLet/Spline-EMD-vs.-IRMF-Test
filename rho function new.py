import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# u range
u = np.linspace(-4, 4, 1000)
H = 1

# Mean loss
rho_mean = u**2 / 2

# Median loss
rho_median = np.abs(u)

# Smoothed median loss
rho_smooth = (
    u * erf(u / (np.sqrt(2) * H))
    + H * np.sqrt(2 / np.pi) * np.exp(-u**2 / (2 * H**2))
)

plt.figure(figsize=(7, 4.5))

# Blue parabola
plt.plot(
    u, rho_mean,
    color='#1f77b4',
    linewidth=3
)

# Red |u|
plt.plot(
    u, rho_median,
    color='red',
    linewidth=3
)

# Green dashed curve
plt.plot(
    u, rho_smooth,
    color='#2ca02c',
    linestyle='--',
    linewidth=3
)

plt.xlabel(r'$u$', fontsize=18)
plt.ylabel(r'$\rho(u)$', fontsize=18)

plt.xlim(-4, 4)
plt.ylim(0, 8.2)

# 与PPT更接近
plt.xticks(np.arange(-4, 5, 1), fontsize=14)
plt.yticks(np.arange(0, 9, 1), fontsize=14)

# 不显示上边框和右边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()