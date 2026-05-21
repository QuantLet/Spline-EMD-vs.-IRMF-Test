#!/usr/bin/python
# coding: UTF-8

import itertools
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ============================================================
# Save Output (h1 × a × hmin)
# ============================================================
OUTPUT_DIR = Path("IRMF Signal 分解 1Hz + 3Hz + 9Hz plots ")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True


# ============================================================
# 1. Epanechnikov 核与三阶样条鲁棒损失函数
# ============================================================
def epanechnikov_kernel(u):
    abs_u = np.abs(u)
    kernel_val = 0.75 * (1.0 - abs_u) ** 2
    kernel_val[abs_u >= 1.0] = 0.0
    return kernel_val


def robust_rho_with_grad(x, H):
    abs_x = np.abs(x)
    sgn = np.sign(x)
    val = np.zeros_like(x)
    grad = np.zeros_like(x)

    # 核心信任区：纯平方误差
    idx1 = abs_x <= H
    val[idx1] = 0.5 * x[idx1] ** 2
    grad[idx1] = x[idx1]

    # 平滑过渡区：三阶样条微调
    idx2 = (abs_x > H) & (abs_x <= 2 * H)
    ax2 = abs_x[idx2]
    val[idx2] = (0.5 * ax2 ** 2) - (1.0 / 6.0) * (ax2 - H) ** 3
    grad[idx2] = x[idx2] - 0.5 * sgn[idx2] * (ax2 - H) ** 2

    # 强力抗噪区：刚性线性截断
    idx3 = abs_x > 2 * H
    ax3 = abs_x[idx3]
    val[idx3] = (0.5 * (2 * H) ** 2 - (1.0 / 6.0) * H ** 3) + 1.5 * H * (ax3 - 2 * H)
    grad[idx3] = sgn[idx3] * 1.5 * H

    return val, grad


# ============================================================
# 2. 周期延拓（显式 3倍 Wrap）IRMF 核心分解算法
# ============================================================
def irmf_academic_decomposition(Y, h1=0.15, a=np.sqrt(2), h_min=0.008, H=0.85):
    n = len(Y)
    t = np.linspace(0, 1, n, endpoint=False)
    Y_k = Y.copy()
    h = h1
    imfs = []
    residuals_history = []

    # 💡 加上 1e-9 容错
    while h >= (h_min - 1e-9):
        S_k = np.zeros(n)

        # 周期信号延拓
        Y_k_extended = np.concatenate([Y_k, Y_k, Y_k])
        t_extended = np.concatenate([t - 1.0, t, t + 1.0])

        for t_idx in range(n):
            t_curr = t[t_idx]
            u_diff = t_curr - t_extended

            # 计算核权重
            kh = epanechnikov_kernel(u_diff / h)
            valid_idx = kh > 0
            if not np.any(valid_idx):
                continue

            kh_valid = kh[valid_idx]
            Y_k_valid = Y_k_extended[valid_idx]

            # 鲁棒凸优化求解
            def objective_and_grad(x):
                loss_vals, grad_vals = robust_rho_with_grad(Y_k_valid - x, H)
                total_loss = np.sum(loss_vals * kh_valid)
                total_grad = np.sum(-grad_vals * kh_valid)
                return total_loss, total_grad

            x0 = Y_k[t_idx]
            res = minimize(objective_and_grad, x0, jac=True, method="L-BFGS-B")
            S_k[t_idx] = res.x[0]

        imfs.append(S_k)
        Y_k = Y_k - S_k
        residuals_history.append(Y_k.copy())
        h = h / a

    return imfs, Y_k, residuals_history


# ============================================================
# 3. 双纵列Plots
# ============================================================
def plot_academic_results_with_residuals_adaptive(
    t,
    signal_Y,
    clean_signal,
    imfs,
    residuals_history,
    final_residual,
    noise,
    title_str,
    filename,
):
    k_total = len(imfs)

    #
    if k_total == 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 2.5), facecolor="none")
        fig.patch.set_alpha(0)
        fig.suptitle(f"{title_str} (No IMFs)", fontsize=12, fontweight="bold")
        axes[0].plot(t, signal_Y, "r", alpha=0.5, label="Observed")
        axes[0].plot(t, clean_signal, "k--")
        axes[1].plot(t, signal_Y, "gray")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300, transparent=True)
        plt.close()
        return

    fig, axes = plt.subplots(
        k_total + 1, 2, figsize=(15, 1.8 * (k_total + 1)), facecolor="none"
    )
    fig.patch.set_alpha(0)
    fig.suptitle(title_str, fontsize=13, fontweight="bold")

    # --- 第一行：原始输入信号形态 ---
    axes[0, 0].plot(t, signal_Y, "r", alpha=0.5, label="Observed Signal $Y(t)$")
    axes[0, 0].plot(t, clean_signal, "k--", label="Clean Signal")
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot(t, signal_Y, "gray", alpha=0.7, label="Initial Residual $Y^{(1)}$")
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # --- 中间行：逐层 IMF 与 对应最新残差演化 ---
    for i in range(k_total):
        axes[i + 1, 0].plot(t, imfs[i], "b", label=f"IMF {i + 1}")
        axes[i + 1, 0].set_xlim([0, 1])
        axes[i + 1, 0].grid(True, linestyle=":", alpha=0.6)

        if i < k_total - 1:
            axes[i + 1, 1].plot(
                t, residuals_history[i], "orange", label=f"Residual $Y^{{({i + 2})}}$"
            )
        else:
            axes[i + 1, 1].plot(
                t, final_residual, "g", linewidth=1.2, label="Final Residual"
            )
            axes[i + 1, 1].plot(t, noise, "k:", alpha=0.4, label="True Noise")

        axes[i + 1, 1].set_xlim([0, 1])
        axes[i + 1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0.085, 0, 1, 0.95])
    # 净化路径文件名
    clean_filename = filename.replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / f"{clean_filename}.png", dpi=300, transparent=True)
    plt.close()


# ============================================================
# 4. Main 主仿真控制流
# ============================================================
if __name__ == "__main__":
    n = 500
    t = np.linspace(0, 1, n, endpoint=False)

    # 目标测试信号 (1Hz + 3Hz + 9Hz 固有谐波)
    clean_signal = (
        1.2 * np.sin(2 * np.pi * t)
        + 0.5 * np.sin(6 * np.pi * t)
        + 0.25 * np.sin(18 * np.pi * t)
    )

    np.random.seed(42)
    base_noise = np.random.normal(0, 1, n)

    # 网格测试空间参数化组合
    h1_options = [0.12, 0.15, 0.18]  # 初始带宽搜索项
    a_options = [np.sqrt(2), 2.0]  # 步进因子搜索项
    h_min_options = [0.005, 0.012]  # 终止边界搜索项

    # 使用卡氏积自动交叉生成全维网格点 (3 * 2 * 2 = 12组)
    grid_points = list(itertools.product(h1_options, a_options, h_min_options))
    total_runs = len(grid_points)

    summary_reports = []

    print(f"========================================================")
    print(
        f" 正在计算并绘制所有 {total_runs} 组超参数图像..."
    )
    print(f"========================================================")

    for idx, (h1_v, a_v, hmin_v) in enumerate(grid_points):
        # 格式化 a 的显示标签
        a_label = "sqrt(2)" if np.isclose(a_v, np.sqrt(2)) else f"{a_v:.1f}"
        print(
            f" 正在处理 组合 [{idx + 1}/{total_runs}]: h1={h1_v:.2f} | a={a_label} | h_min={hmin_v:.4f}"
        )

        # --------------------------------------------------------
        # 🧪 实验一：0.05 弱噪声组
        # --------------------------------------------------------
        noise_005 = 0.05 * base_noise
        signal_Y_005 = clean_signal + noise_005
        imfs_005, final_res_005, res_hist_005 = irmf_academic_decomposition(
            signal_Y_005, h1=h1_v, a=a_v, h_min=hmin_v, H=0.60
        )

        # 弱噪组图像
        plot_academic_results_with_residuals_adaptive(
            t,
            signal_Y_005,
            clean_signal,
            imfs_005,
            res_hist_005,
            final_res_005,
            noise_005,
            f"Weak Noise Track (h1={h1_v:.2f}, a={a_label}, h_min={hmin_v:.4f})",
            f"weak_noise_run_{idx + 1}_h1_{h1_v:.2f}_a_{a_label}_hmin_{hmin_v:.4f}",
        )

        # --------------------------------------------------------
        # 🧪 实验二：0.20 强噪声组
        # --------------------------------------------------------
        noise_020 = 0.2 * base_noise
        signal_Y_020 = clean_signal + noise_020
        imfs_020, final_res_020, res_hist_020 = irmf_academic_decomposition(
            signal_Y_020, h1=h1_v, a=a_v, h_min=hmin_v, H=0.85
        )

        # 强噪组图像
        plot_academic_results_with_residuals_adaptive(
            t,
            signal_Y_020,
            clean_signal,
            imfs_020,
            res_hist_020,
            final_res_020,
            noise_020,
            f"Strong Noise Track (h1={h1_v:.2f}, a={a_label}, h_min={hmin_v:.4f})",
            f"strong_noise_run_{idx + 1}_h1_{h1_v:.2f}_a_{a_label}_hmin_{hmin_v:.4f}",
        )

        # 记录层数
        summary_reports.append(
            {
                "h1": h1_v,
                "a_str": a_label,
                "hmin": hmin_v,
                "weak_layers": len(imfs_005),
                "strong_layers": len(imfs_020),
            }
        )

    # ============================================================
    # 纯模态分解层数多维敏感性分析表
    # ============================================================
    print("\n\n" + "=" * 55)
    print("【IRMF 全参数空间 (h1 × a × h_min) 模态分解层数对比大表】")
    print("=" * 55)
    print(f"{'h1':<5} | {'a':<8} | {'h_min':<7} | {'弱噪分量层数':<12} | {'强噪分量层数'}")
    print("-" * 55)

    for item in summary_reports:
        print(
            f"{item['h1']:<5.2f} | "
            f"{item['a_str']:<8} | "
            f"{item['hmin']:<7.4f} | "
            f"{item['weak_layers']:^14} | "
            f"{item['strong_layers']:^12}"
        )

    print("-" * 55)
    print(
        f"全空间 12 组网格扫描与 24 张高清图谱已成功完成！见目录：./{OUTPUT_DIR}/"
    )