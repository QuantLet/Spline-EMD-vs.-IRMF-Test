#!/usr/bin/python
# coding: UTF-8

"""Methodology layer: loss-function analysis for the V4E paper."""

from math import erf
import numpy as np

from core_algorithms.strict_spokoiny_irmf import rho_spline_grad_hess
from experiments.paper_pipeline_utils import ensure_dir, write_csv, write_json


def _normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def _l2(x):
    return 0.5 * x ** 2, x, np.ones_like(x)


def _l1(x):
    return np.abs(x), np.sign(x), np.zeros_like(x)


def _huber(x, delta=1.0):
    ax = np.abs(x)
    rho = np.where(ax <= delta, 0.5 * x ** 2, delta * (ax - 0.5 * delta))
    grad = np.where(ax <= delta, x, delta * np.sign(x))
    hess = np.where(ax <= delta, 1.0, 0.0)
    return rho, grad, hess


def _pseudo_huber(x, delta=1.0):
    z = x / delta
    rho = delta ** 2 * (np.sqrt(1.0 + z ** 2) - 1.0)
    grad = x / np.sqrt(1.0 + z ** 2)
    hess = (1.0 + z ** 2) ** (-1.5)
    return rho, grad, hess


def _smoothed_median_closed_form(x, H=1.0):
    rho, grad, hess = rho_spline_grad_hess(x, H)
    return rho, grad, hess


def _summarize_loss(name, x, rho, grad, hess):
    finite_hess = hess[np.isfinite(hess)]
    positive_hess_ratio = float(np.mean(finite_hess >= -1e-10)) if finite_hess.size else None
    tail = np.abs(x) >= 3.0
    center = np.abs(x) <= 0.25
    return {
        "loss": name,
        "max_abs_influence_on_grid": float(np.nanmax(np.abs(grad))),
        "tail_mean_abs_influence": float(np.nanmean(np.abs(grad[tail]))) if np.any(tail) else None,
        "center_mean_curvature": float(np.nanmean(hess[center])) if np.any(center) else None,
        "max_curvature_on_grid": float(np.nanmax(hess)),
        "positive_hessian_ratio": positive_hess_ratio,
        "rho_at_zero": float(rho[np.argmin(np.abs(x))]),
        "gradient_at_zero": float(grad[np.argmin(np.abs(x))]),
    }


def _write_loss_figures(rows, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json({"figure_status": "skipped", "reason": str(exc)}, output_dir / "loss_figures_status.json")
        return

    x = np.asarray([r["x"] for r in rows if r["loss"] == "Gaussian-smoothed median"], dtype=float)
    by_loss = {}
    for row in rows:
        by_loss.setdefault(row["loss"], []).append(row)

    styles = {
        "L2 mean": {"color": "#2B6CB0", "linestyle": "-"},
        "L1 median": {"color": "#2F855A", "linestyle": "-"},
        "Huber": {"color": "#C05621", "linestyle": "--"},
        "Pseudo-Huber": {"color": "#805AD5", "linestyle": "-."},
        "Gaussian-smoothed median": {"color": "#C53030", "linestyle": "-"},
    }
    for field, ylabel, fname in [
        ("rho", "loss", "loss_curves.png"),
        ("gradient", "score / influence", "influence_curves.png"),
        ("hessian", "second derivative", "curvature_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        for loss_name, loss_rows in by_loss.items():
            ys = np.asarray([r[field] for r in loss_rows], dtype=float)
            st = styles.get(loss_name, {})
            ax.plot(x, ys, label=loss_name, linewidth=2.0, **st)
        ax.axhline(0, color="#444444", linewidth=0.7)
        ax.axvline(0, color="#444444", linewidth=0.7)
        ax.set_xlabel("residual")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / fname)
        plt.close(fig)


def run_loss_function_analysis(output_root, H=1.0, x_min=-5.0, x_max=5.0, n_grid=1001):
    """
    Compare the V4E Gaussian-smoothed median loss with standard robust losses.

    Outputs are intended for the methodology section and Appendix B:
    loss/score/curvature curves, influence summaries, and proof-ready property
    checks for convexity, bounded score, and smoothness.
    """
    output_root = ensure_dir(output_root)
    x = np.linspace(float(x_min), float(x_max), int(n_grid))
    losses = {
        "L2 mean": _l2(x),
        "L1 median": _l1(x),
        "Huber": _huber(x, delta=1.0),
        "Pseudo-Huber": _pseudo_huber(x, delta=1.0),
        "Gaussian-smoothed median": _smoothed_median_closed_form(x, H=H),
    }

    rows = []
    summary_rows = []
    for loss_name, (rho, grad, hess) in losses.items():
        for xi, ri, gi, hi in zip(x, rho, grad, hess):
            rows.append({
                "x": float(xi),
                "loss": loss_name,
                "rho": float(ri),
                "gradient": float(gi),
                "hessian": float(hi),
            })
        summary_rows.append(_summarize_loss(loss_name, x, rho, grad, hess))

    sm_rho, sm_grad, sm_hess = losses["Gaussian-smoothed median"]
    cdf_grad_identity = 2.0 * _normal_cdf(x / H) - 1.0
    property_checks = {
        "H": float(H),
        "convexity_grid_check_min_hessian": float(np.min(sm_hess)),
        "convexity_grid_check_pass": bool(np.min(sm_hess) >= -1e-10),
        "bounded_influence_max_abs_gradient": float(np.max(np.abs(sm_grad))),
        "bounded_influence_pass": bool(np.max(np.abs(sm_grad)) <= 1.0 + 1e-10),
        "even_loss_check_max_abs_difference": float(np.max(np.abs(sm_rho - sm_rho[::-1]))),
        "odd_gradient_check_max_abs_sum": float(np.max(np.abs(sm_grad + sm_grad[::-1]))),
        "closed_form_gradient_identity_max_abs_error": float(np.max(np.abs(sm_grad - cdf_grad_identity))),
        "curvature_at_zero": float(sm_hess[np.argmin(np.abs(x))]),
    }

    proof_notes = """# Proof-ready loss properties

Proposition 1 (convexity and smoothness).  For H > 0, the Gaussian-smoothed
median loss

    rho_H(x) = sqrt(2/pi) H exp(-x^2/(2H^2)) + x {2 Phi(x/H) - 1}

is continuously differentiable and convex because

    rho'_H(x) = 2 Phi(x/H) - 1,
    rho''_H(x) = (2/H) phi(x/H) >= 0.

Proposition 2 (bounded influence).  The score rho'_H(x) lies in [-1, 1] for all
x because Phi(.) lies in [0, 1].  Therefore a single residual has bounded
pointwise influence in the local M-estimation equation.

These are mathematical properties of the loss itself.  They should be stated
separately from empirical Spokoiny-inspired diagnostics, which validate the
implemented IRMF pipeline rather than proving a full non-asymptotic theorem.
"""

    write_csv(rows, output_root / "loss_function_curves.csv")
    write_json(rows, output_root / "loss_function_curves.json")
    write_csv(summary_rows, output_root / "loss_function_summary.csv")
    write_json(summary_rows, output_root / "loss_function_summary.json")
    write_json(property_checks, output_root / "smoothed_median_property_checks.json")
    (output_root / "proof_ready_loss_properties.md").write_text(proof_notes, encoding="utf-8")
    _write_loss_figures(rows, output_root)

    return {
        "n_grid": int(n_grid),
        "losses_compared": list(losses.keys()),
        "smoothed_median_property_checks": property_checks,
        "summary_rows": summary_rows,
    }

