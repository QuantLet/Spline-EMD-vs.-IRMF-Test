#!/usr/bin/python
# coding: UTF-8

import os
from pathlib import Path
import itertools
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# Save Output
# ============================================================
OUTPUT_DIR = Path("IRMF Signal Evaluation for 1Hz + 3Hz + 9Hz plots")
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

    # 核心信任区
    idx1 = abs_x <= H
    val[idx1] = 0.5 * x[idx1] ** 2
    grad[idx1] = x[idx1]

    # 平滑过渡区
    idx2 = (abs_x > H) & (abs_x <= 2 * H)
    ax2 = abs_x[idx2]
    val[idx2] = (0.5 * ax2 ** 2) - (1.0 / 6.0) * (ax2 - H) ** 3
    grad[idx2] = x[idx2] - 0.5 * sgn[idx2] * (ax2 - H) ** 2

    # 强力抗噪区
    idx3 = abs_x > 2 * H
    ax3 = abs_x[idx3]
    val[idx3] = (0.5 * (2 * H) ** 2 - (1.0 / 6.0) * H ** 3) + 1.5 * H * (ax3 - 2 * H)
    grad[idx3] = sgn[idx3] * 1.5 * H

    return val, grad


# ============================================================
# 2. 周期延拓（Periodic Extension）IRMF 算法
# ============================================================
def irmf_academic_decomposition(Y, h1=0.15, a=np.sqrt(2), h_min=0.008, H=0.85):
    n = len(Y)
    t = np.linspace(0, 1, n, endpoint=False)
    Y_k = Y.copy()
    h = h1
    imfs = []
    residuals_history = []

    while h >= h_min:
        S_k = np.zeros(n)
        Y_k_extended = np.concatenate([Y_k, Y_k, Y_k])
        t_extended = np.concatenate([t - 1.0, t, t + 1.0])

        for t_idx in range(n):
            t_curr = t[t_idx]
            u_diff = t_curr - t_extended

            kh = epanechnikov_kernel(u_diff / h)
            valid_idx = kh > 0
            if not np.any(valid_idx):
                continue

            kh_valid = kh[valid_idx]
            Y_k_valid = Y_k_extended[valid_idx]

            def objective_and_grad(x):
                loss_vals, grad_vals = robust_rho_with_grad(Y_k_valid - x, H)
                total_loss = np.sum(loss_vals * kh_valid)
                total_grad = np.sum(-grad_vals * kh_valid)
                return total_loss, total_grad

            x0 = Y_k[t_idx]
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B')
            S_k[t_idx] = res.x[0]

        imfs.append(S_k)
        Y_k = Y_k - S_k
        residuals_history.append(Y_k.copy())
        h = h / a

    return imfs, Y_k, residuals_history


# ============================================================
# 3. 模型评估 Part 1
# ============================================================
def evaluate_decomposition_performance_irmf(Y_observed, X_clean, imfs, final_res, fs):
    imfNo = len(imfs)
    N = len(Y_observed)

    if imfNo == 0:
        return {
            "imf_count": 0, "io": 999.0, "center_freqs": [],
            "denoise_mse": 999.0, "denoise_psnr": -99.0, "denoise_corr": 0.0,
            "completeness_mse": 999.0, "completeness_psnr": -99.0, "completeness_corr": 0.0,
            "spectral_corr": 0.0, "energy_ratio": 0.0, "max_leakage": 1.0
        }

    imfs_arr = np.array(imfs)
    peak_val_clean = np.max(X_clean) - np.min(X_clean)
    peak_val_obs = np.max(Y_observed) - np.min(Y_observed)

    # --------------------------------------------------------
    # 轨 1：信号去噪/重构精度评估轨道 (剔除噪声残差，对比纯净信号 X_clean)
    # --------------------------------------------------------
    X_recon_denoise = np.sum(imfs_arr, axis=0)
    denoise_mse = np.mean((X_recon_denoise - X_clean) ** 2)
    denoise_psnr = 10 * np.log10((peak_val_clean ** 2) / (denoise_mse + 1e-10))
    denoise_corr = np.corrcoef(X_recon_denoise, X_clean)[0, 1] if np.std(X_recon_denoise) > 0 else 0.0

    # --------------------------------------------------------
    # 轨 2：代数完备性验证轨道 (包含残差，对比含噪原始输入 Y_observed)
    # --------------------------------------------------------
    X_recon_complete = np.sum(imfs_arr, axis=0) + final_res
    complete_mse = np.mean((X_recon_complete - Y_observed) ** 2)
    complete_psnr = 10 * np.log10((peak_val_obs ** 2) / (complete_mse + 1e-10))
    complete_corr = 1.0 if np.allclose(X_recon_complete, Y_observed) else np.corrcoef(X_recon_complete, Y_observed)[
        0, 1]

    # --------------------------------------------------------
    # 基础数理特性：全局正交指数 IO 与 频域质心 CF
    # --------------------------------------------------------
    all_components = list(imfs) + [final_res]
    total_components = len(all_components)
    numerator = 0.0
    for i in range(total_components):
        for j in range(total_components):
            if i != j:
                numerator += np.sum(all_components[i] * all_components[j])
    denominator = np.sum(Y_observed ** 2)
    io_index = numerator / (denominator + 1e-10)

    center_freqs = []
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    for i in range(imfNo):
        imf_fft = np.fft.rfft(imfs[i])
        psd = np.abs(imf_fft) ** 2
        cf = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        center_freqs.append(cf)

    # --------------------------------------------------------
    # 模型评估 Part 2
    # --------------------------------------------------------
    # 1. 频域功率谱互相关系数 (Spectral Pearson Correlation) -> 验证频域吻合度
    psd_recon = np.abs(np.fft.rfft(X_recon_denoise)) ** 2
    psd_clean = np.abs(np.fft.rfft(X_clean)) ** 2
    if np.std(psd_recon) > 0 and np.std(psd_clean) > 0:
        spectral_corr = np.corrcoef(psd_recon, psd_clean)[0, 1]
    else:
        spectral_corr = 0.0

    # 2. 能量守恒百分比 (Energy Conservation Ratio, ECR) -> 比单纯误差更直观展示 100% 守恒
    energy_components = np.sum([np.sum(c ** 2) for c in all_components])
    energy_observed = np.sum(Y_observed ** 2)
    energy_ratio = (energy_components / (energy_observed + 1e-10)) * 100.0

    # 3. 相邻模态最大泄露指数 (Max Adjacent Mode Leakage) -> 定量控诉混叠
    max_leakage = 0.0
    if imfNo > 1:
        for i in range(imfNo - 1):
            imf1, imf2 = imfs[i], imfs[i + 1]
            if np.std(imf1) > 0 and np.std(imf2) > 0:
                leakage = np.abs(np.corrcoef(imf1, imf2)[0, 1])
                if leakage > max_leakage:
                    max_leakage = leakage

    return {
        "imf_count": imfNo, "io": io_index, "center_freqs": center_freqs,
        "denoise_mse": denoise_mse, "denoise_psnr": denoise_psnr, "denoise_corr": denoise_corr,
        "completeness_mse": complete_mse, "completeness_psnr": complete_psnr, "completeness_corr": complete_corr,
        "spectral_corr": spectral_corr,
        "energy_ratio": energy_ratio,
        "max_leakage": max_leakage
    }


# ============================================================
# 4. 多尺度滤波器组特性图导出
# ============================================================
def export_spectral_overlap_chart_irmf(imfs, fs, noise_label, file_prefix, h1, a_label, hmin):
    if len(imfs) == 0: return
    imfNo = len(imfs)
    N = len(imfs[0])
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    plt.figure(figsize=(10, 4))
    for i in range(imfNo):
        imf_fft = np.fft.rfft(imfs[i])
        psd = np.abs(imf_fft) ** 2
        psd_norm = psd / (np.max(psd) + 1e-10)
        plt.plot(freqs, psd_norm, label=f"IMF {i + 1}", alpha=0.75, linewidth=1.5)

    plt.title(f"Filter Bank (IRMF at {noise_label} Noise | h1={h1:.2f}, a={a_label}, h_min={hmin:.4f})", fontsize=11,
              fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Normalized PSD", fontsize=10)
    plt.xlim(0, fs / 2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_prefix}_spectral_overlap.png", dpi=300, transparent=True)
    plt.close()


# ============================================================
# 5. Main 主网格遍历控制流
# ============================================================
if __name__ == "__main__":
    n = 500
    fs = 500.0
    t = np.linspace(0, 1, n, endpoint=False)

    np.random.seed(0)
    varepsilon = np.random.randn(n)

    # 1Hz + 3Hz + 9Hz 标准多谐波测试信号
    X_clean = 1.2 * np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t) + 0.25 * np.sin(18 * np.pi * t)

    h1_options = [0.12, 0.15, 0.18]
    a_options = [np.sqrt(2), 2.0]
    h_min_options = [0.005, 0.012]

    grid_combinations = list(itertools.product(h1_options, a_options, h_min_options))
    total_runs = len(grid_combinations)
    grid_summary_results = []

    print(f"========================================================")
    print(f" 正在扫描 3D 超参数空间，共计 {total_runs} 组超参数组合...")
    print(f"========================================================")

    for idx, (h1_v, a_v, hmin_v) in enumerate(grid_combinations):
        a_name = "sqrt(2)" if np.isclose(a_v, np.sqrt(2)) else f"{a_v:.1f}"
        print(f" 正在评测 组合 [{idx + 1}/{total_runs}]: h1={h1_v:.2f} | a={a_name} | h_min={hmin_v:.4f}")

        # 🧪 实验一：0.05 弱噪组测试 (自适应 H=0.60)
        noise_005 = 0.05 * varepsilon
        Y_005 = X_clean + noise_005
        imfs_005, final_res_005, _ = irmf_academic_decomposition(Y_005, h1=h1_v, a=a_v, h_min=hmin_v, H=0.60)
        metrics_005 = evaluate_decomposition_performance_irmf(Y_005, X_clean, imfs_005, final_res_005, fs)

        file_pfx_005 = f"run_{idx + 1}_weak_h1_{h1_v:.2f}_a_{a_name}_hmin_{hmin_v:.4f}"
        export_spectral_overlap_chart_irmf(imfs_005, fs, "Weak", file_pfx_005, h1_v, a_name, hmin_v)

        # 🧪 实验二：0.20 强噪组测试 (自适应 H=0.85)
        noise_020 = 0.2 * varepsilon
        Y_020 = X_clean + noise_020
        imfs_020, final_res_020, _ = irmf_academic_decomposition(Y_020, h1=h1_v, a=a_v, h_min=hmin_v, H=0.85)
        metrics_020 = evaluate_decomposition_performance_irmf(Y_020, X_clean, imfs_020, final_res_020, fs)

        file_pfx_020 = f"run_{idx + 1}_strong_h1_{h1_v:.2f}_a_{a_name}_hmin_{hmin_v:.4f}"
        export_spectral_overlap_chart_irmf(imfs_020, fs, "Strong", file_pfx_020, h1_v, a_name, hmin_v)

        grid_summary_results.append({
            "h1": h1_v,
            "a_str": a_name,
            "hmin": hmin_v,
            "m_005": metrics_005,
            "m_020": metrics_020
        })

    # ============================================================
    # 表 1：时频域净化重构质量 (剔除残差对比 X_clean | 新增频域相关 Rf 及最大模态泄漏 MEL)
    # ============================================================
    print("\n\n" + "#" * 135)
    print(" 【表1：IRMF 多维参数空间 [时频双轨净化重构质量] (剔除噪声残差对比纯净信号 X_clean)】")
    print("#" * 135)
    print(
        f"{'h1':<5} | {'a':<8} | {'h_min':<7} | {'噪声环境':<8} | {'层数':<4} | {'去噪 MSE':<12} | {'去噪 PSNR':<11} | {'时域相关 R':<10} | {'频域相关 Rf':<11} | {'相邻最大泄漏':<12} | {'正交指数 IO'}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]
        print(
            f"{item['h1']:<5.2f} | {item['a_str']:<8} | {item['hmin']:<7.4f} | {'0.05弱噪':<8} | {m5['imf_count']:^4} | {m5['denoise_mse']:>12.6f} | {m5['denoise_psnr']:>8.2f} dB  | {m5['denoise_corr']:>10.4f} | {m5['spectral_corr']:>11.4f} | {m5['max_leakage']:>12.4f} | {m5['io']:>11.6f}")
        print(
            f"{'':<5} | {'':<8} | {'':<7} | {'0.20强噪':<8} | {m20['imf_count']:^4} | {m20['denoise_mse']:>12.6f} | {m20['denoise_psnr']:>8.2f} dB  | {m20['denoise_corr']:>10.4f} | {m20['spectral_corr']:>11.4f} | {m20['max_leakage']:>12.4f} | {m20['io']:>11.6f}")
        print("-" * 135)

    # ============================================================
    # 表 2：代数完备性与能量守恒验证 (包含残差对比 Y_observed | 新增能量守恒率 ECR)
    # ============================================================
    print("\n\n" + "#" * 135)
    print(" 【表2：IRMF 算法代数无损完备性与能量守恒自证 (包含残差对比原始输入 Y_observed)】")
    print("#" * 135)
    print(
        f"{'h1':<5} | {'a':<8} | {'h_min':<7} | {'噪声环境':<8} | {'完备性 MSE':<14} | {'完备性 PSNR':<14} | {'完备时域相关 R':<15} | {'能量守恒率 (ECR, %)'}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]
        print(
            f"{item['h1']:<5.2f} | {item['a_str']:<8} | {item['hmin']:<7.4f} | {'0.05弱噪':<8} | {m5['completeness_mse']:>14.2e} | {m5['completeness_psnr']:>11.2f} dB  | {m5['completeness_corr']:>15.4f} | {m5['energy_ratio']:>18.6f} %")
        print(
            f"{'':<5} | {'':<8} | {'':<7} | {'0.20强噪':<8} | {m20['completeness_mse']:>14.2e} | {m20['completeness_psnr']:>11.2f} dB  | {m20['completeness_corr']:>15.4f} | {m20['energy_ratio']:>18.6f} %")
        print("-" * 135)

    # ============================================================
    # 表 3：频域滤波器群特征分布
    # ============================================================
    print("\n\n" + "#" * 135)
    print("📋 【表3：IRMF 滤波器组频域阶梯演化特性测试（各层 IMF 频谱质心 Center Frequency）】")
    print("#" * 135)
    print(
        f"{'h1':<4} | {'a':<8} | {'h_min':<7} | {'噪声环境':<6} | {'各层 IMF 频谱质心分布 (Center Frequency, 单位: Hz)':<40}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]

        cf_005_str = " -> ".join([f"IMF{i + 1}:{cf:.2f}Hz" for i, cf in enumerate(m5["center_freqs"])])
        cf_020_str = " -> ".join([f"IMF{i + 1}:{cf:.2f}Hz" for i, cf in enumerate(m20["center_freqs"])])

        print(f"{item['h1']:<4.2f} | {item['a_str']:<8} | {item['hmin']:<7.4f} | {'0.05弱':<6} | {cf_005_str}")
        print(f"{'':<4} | {'':<8} | {'':<7} | {'0.20强':<6} | {cf_020_str}")
        print("-" * 135)

    print(
        f"\n 图谱已输出至 ./{OUTPUT_DIR}/")