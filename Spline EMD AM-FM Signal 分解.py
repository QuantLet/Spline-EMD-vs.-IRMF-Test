#!/usr/bin/python
# coding: UTF-8

import itertools
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PyEMD import EMD

# ============================================================
# Output settings
# ============================================================

OUTPUT_DIR = Path("Spline EMD Signal 分解 AM-FM 调制信号 updated Results")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True


def _safe_filename(title):
    safe = "".join(char.lower() if char.isalnum() else "_" for char in title)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:160] or "plot"


def save_current_plot(title):
    output_path = OUTPUT_DIR / f"{_safe_filename(title)}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved figure: {output_path}")
    plt.close()


# ============================================================
# 评估体系
# ============================================================
def evaluate_decomposition_performance_emd(Y_observed, X_clean, imfs, final_res, fs):
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
    # 其他指标
    # --------------------------------------------------------
    # 1. 频域功率谱互相关系数 (Spectral Pearson Correlation)
    psd_recon = np.abs(np.fft.rfft(X_recon_denoise)) ** 2
    psd_clean = np.abs(np.fft.rfft(X_clean)) ** 2
    if np.std(psd_recon) > 0 and np.std(psd_clean) > 0:
        spectral_corr = np.corrcoef(psd_recon, psd_clean)[0, 1]
    else:
        spectral_corr = 0.0

    # 2. 能量守恒百分比 (Energy Conservation Ratio, ECR)
    energy_components = np.sum([np.sum(c ** 2) for c in all_components])
    energy_observed = np.sum(Y_observed ** 2)
    energy_ratio = (energy_components / (energy_observed + 1e-10)) * 100.0

    # 3. 相邻模态最大泄露指数 (Max Adjacent Mode Leakage)
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
# 多尺度滤波器组特性可视化导出 (Filter Bank Chart)
# ============================================================
def export_spectral_overlap_chart_emd(imfs, fs, noise_label, file_prefix):
    if len(imfs) == 0:
        return
    imfNo = len(imfs)
    N = len(imfs[0])
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    plt.figure(figsize=(10, 4))
    for i in range(imfNo):
        imf_fft = np.fft.rfft(imfs[i])
        psd = np.abs(imf_fft) ** 2
        psd_norm = psd / (np.max(psd) + 1e-10)
        plt.plot(freqs, psd_norm, label=f"IMF {i + 1}", alpha=0.75, linewidth=1.5)

    plt.title(f"Filter Bank (EMD Benchmark at {noise_label} Noise)", fontsize=11, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Normalized PSD", fontsize=10)
    plt.xlim(0, fs / 2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    plt.tight_layout()

    save_current_plot(f"{file_prefix}_spectral_overlap")


# ============================================================
# Main 主仿真流
# ============================================================
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    n = 500
    fs = 500.0
    tMin, tMax = 0.0, 1.0
    T = np.linspace(tMin, tMax, n, endpoint=False, dtype=np.float64)

    np.random.seed(0)
    varepsilon = np.random.randn(n)

    # ✨ 核心修改点：替换为 AM-FM 宽带非平稳调制测试信号真值
    X_clean = (1 + 0.5 * np.sin(4 * np.pi * T)) * np.sin(2 * np.pi * (6 * T + 12 * T ** 2))

    grid_summary_results = []

    print(f"========================================================")
    print(f"正在进行双噪声场景自适应分解 (AM-FM 调制信号)...")
    print(f"========================================================")

    # 实验一：0.05 弱噪组测试
    print(" 正在评测 EMD 弱噪基准组: sigma=0.05")
    noise_005 = 0.05 * varepsilon
    Y_005 = X_clean + noise_005

    emd_005 = EMD()
    emd_005.FIXE_H = 5
    emd_005.nbsym = 2
    emd_005.spline_kind = "cubic"
    emd_005.DTYPE = np.float64

    all_comps_005 = emd_005.emd(Y_005, T, -1)
    imfs_005 = all_comps_005[:-1]
    final_res_005 = all_comps_005[-1]
    metrics_005 = evaluate_decomposition_performance_emd(Y_005, X_clean, imfs_005, final_res_005, fs)

    # 导出 EMD 在弱噪环境下的 Filter Bank 功率谱特征图
    export_spectral_overlap_chart_emd(imfs_005, fs, "Weak", "emd_run_weak")

    # 实验二：0.20 强噪组测试
    print(" 正在评测 EMD 强噪基准组: sigma=0.20")
    noise_020 = 0.2 * varepsilon
    Y_020 = X_clean + noise_020

    emd_020 = EMD()
    emd_020.FIXE_H = 5
    emd_020.nbsym = 2
    emd_020.spline_kind = "cubic"
    emd_020.DTYPE = np.float64

    all_comps_020 = emd_020.emd(Y_020, T, -1)
    imfs_020 = all_comps_020[:-1]
    final_res_020 = all_comps_020[-1]
    metrics_020 = evaluate_decomposition_performance_emd(Y_020, X_clean, imfs_020, final_res_020, fs)

    # 导出 EMD 在强噪环境下的 Filter Bank 功率谱特征图
    export_spectral_overlap_chart_emd(imfs_020, fs, "Strong", "emd_run_strong")

    # 将双噪声结果结合
    grid_summary_results.append({
        "h1": 0.0,
        "a_str": "Standard",
        "hmin": 0.0,
        "m_005": metrics_005,
        "m_020": metrics_020
    })

    # ============================================================
    # 表 1：时频域净化重构质量
    # ============================================================
    print("\n\n" + "#" * 135)
    print("【表1：EMD 基准算法 [时频双轨净化重构质量表] (剔除噪声残差对比纯净信号 X_clean)】")
    print("#" * 135)
    print(
        f"{'Model':<5} | {'Config':<8} | {'Status':<7} | {'噪声环境':<8} | {'层数':<4} | {'去噪 MSE':<12} | {'去噪 PSNR':<11} | {'时域相关 R':<10} | {'频域相关 Rf':<11} | {'相邻最大泄漏':<12} | {'正交指数 IO'}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]
        print(
            f"{'EMD':<5} | {'Default':<8} | {'Base':<7} | {'0.05弱噪':<8} | {m5['imf_count']:^4} | {m5['denoise_mse']:>12.6f} | {m5['denoise_psnr']:>8.2f} dB  | {m5['denoise_corr']:>10.4f} | {m5['spectral_corr']:>11.4f} | {m5['max_leakage']:>12.4f} | {m5['io']:>11.6f}")
        print(
            f"{'':<5} | {'':<8} | {'':<7} | {'0.20强噪':<8} | {m20['imf_count']:^4} | {m20['denoise_mse']:>12.6f} | {m20['denoise_psnr']:>8.2f} dB  | {m20['denoise_corr']:>10.4f} | {m20['spectral_corr']:>11.4f} | {m20['max_leakage']:>12.4f} | {m20['io']:>11.6f}")
        print("-" * 135)

    # ============================================================
    # 表 2：代数完备性与能量守恒验证
    # ============================================================
    print("\n\n" + "#" * 135)
    print("【表2：EMD 基准算法代数无损完备性与能量守恒自证表 (包含残差对比原始输入 Y_observed)】")
    print("#" * 135)
    print(
        f"{'Model':<5} | {'Config':<8} | {'Status':<7} | {'噪声环境':<8} | {'完备性 MSE':<14} | {'完备性 PSNR':<14} | {'完备时域相关 R':<15} | {'能量守恒率 (ECR, %)'}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]
        print(
            f"{'EMD':<5} | {'Default':<8} | {'Base':<7} | {'0.05弱噪':<8} | {m5['completeness_mse']:>14.2e} | {m5['completeness_psnr']:>11.2f} dB  | {m5['completeness_corr']:>15.4f} | {m5['energy_ratio']:>18.6f} %")
        print(
            f"{'':<5} | {'':<8} | {'':<7} | {'0.20强噪':<8} | {m20['completeness_mse']:>14.2e} | {m20['completeness_psnr']:>11.2f} dB  | {m20['completeness_corr']:>15.4f} | {m20['energy_ratio']:>18.6f} %")
        print("-" * 135)

    # ============================================================
    # 表 3：频域滤波器群特征分布
    # ============================================================
    print("\n\n" + "#" * 135)
    print("【表3：EMD 基准算法各层 IMF 频谱质心 Center Frequency 特性表】")
    print("#" * 135)
    print(
        f"{'Model':<4} | {'Config':<8} | {'Status':<7} | {'噪声环境':<6} | {'各层 IMF 频谱质心分布 (Center Frequency, 单位: Hz)':<40}")
    print("-" * 135)

    for item in grid_summary_results:
        m5 = item["m_005"]
        m20 = item["m_020"]

        cf_005_str = " -> ".join([f"IMF{i + 1}:{cf:.2f}Hz" for i, cf in enumerate(m5["center_freqs"])])
        cf_020_str = " -> ".join([f"IMF{i + 1}:{cf:.2f}Hz" for i, cf in enumerate(m20["center_freqs"])])

        print(f"{'EMD':<4} | {'Default':<8} | {'Base':<7} | {'0.05弱':<6} | {cf_005_str}")
        print(f"{'':<4} | {'':<8} | {'':<7} | {'0.20强':<6} | {cf_020_str}")
        print("-" * 135)

    # ============================================================
    # 绘图
    # ============================================================
    for sigma, imfs_set, final_res_set in [(0.05, imfs_005, final_res_005), (0.20, imfs_020, final_res_020)]:
        Y_plot = X_clean + sigma * varepsilon
        imfNo = len(imfs_set)

        c = 2
        r = imfNo + 1
        fig = plt.figure(figsize=(13, 2.2 * r))
        current_residual = Y_plot.copy()

        # 第一行源头绘制
        plt.subplot(r, c, 1)
        plt.plot(T, Y_plot, "r", linewidth=1.2)
        plt.plot(T, X_clean, "k--", linewidth=1.0)
        plt.xlim((tMin, tMax))
        plt.title(f"EMD Component (Sigma={sigma})", fontsize=11, fontweight='bold', color='darkred')
        plt.ylabel("Input Stage", fontsize=10, fontweight='bold')
        plt.grid(True, linestyle=":", alpha=0.5)

        plt.subplot(r, c, 2)
        plt.plot(T, current_residual, "gray", linewidth=1.2)
        plt.xlim((tMin, tMax))
        plt.title(r"Progressive Residual $\mathcal{R}_k(t)$ (Right)", fontsize=11, fontweight='bold', color='darkblue')
        plt.grid(True, linestyle=":", alpha=0.5)

        # 迭代生成其余中间交互行
        for num in range(imfNo):
            row_idx = num + 2
            imf_curr = imfs_set[num]
            current_residual = current_residual - imf_curr

            plt.subplot(r, c, (row_idx - 1) * 2 + 1)
            plt.plot(T, imf_curr, "g", linewidth=1.2)
            plt.xlim((tMin, tMax))
            plt.ylabel(f"IMF {num + 1}", fontsize=10, fontweight='bold')
            plt.grid(True, linestyle=":", alpha=0.5)

            plt.subplot(r, c, (row_idx - 1) * 2 + 2)
            color_line = "b" if num == imfNo - 1 else "orange"
            line_w = 1.6 if num == imfNo - 1 else 1.2
            plt.plot(T, current_residual, color_line, linewidth=line_w)
            plt.xlim((tMin, tMax))
            plt.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        save_current_plot(f"EMD_comprehensive_progressive_profile_sigma_{sigma}")

    print(
        f"\n Filter Bank 与渐进历史联动图谱已全部导出至 ./{OUTPUT_DIR}/")