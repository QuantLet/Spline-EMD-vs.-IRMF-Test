#!/usr/bin/python
# coding: UTF-8

"""V5.31 time-axis semantics audit and sensitivity-module boundary record."""

from datetime import datetime, timezone

from project_config import DEFAULT_FS, DEFAULT_N, GLOBAL_IRMF_PARAMS
from experiments.paper_pipeline_utils import ensure_dir, write_json


V531_TIME_AXIS_SEMANTICS_VERSION = "V5.31_time_axis_semantics_and_sensitivity_boundary"


def run_v531_time_axis_semantics_audit(output_root):
    output_root = ensure_dir(output_root)
    spec = {
        "schema_version": V531_TIME_AXIS_SEMANTICS_VERSION,
        "audit_status": "passed",
        "design_status": "boundary_recorded",
        "benchmark_results_recomputed": False,
        "algorithm_cube_rerun_required": False,
        "current_synthetic_main_design": {
            "n": int(DEFAULT_N),
            "nominal_fs": float(DEFAULT_FS),
            "time_domain": "normalized_time",
            "signal_time_axis": "t_unit = linspace(0, 1, n, endpoint=False)",
            "physical_like_time_axis_recorded": "T = arange(n) / fs",
            "physical_second_interpretation_allowed": False,
            "interpretation": (
                "n controls sampling density over the normalized interval [0,1), "
                "not physical observation duration."
            ),
        },
        "verified_code_semantics": {
            "synthetic_clean_signal_uses_t_unit": True,
            "synthetic_true_components_use_t_unit": True,
            "case_record_contains_physical_like_T": True,
            "real_ecg_windows_use_native_fs_and_seconds": True,
            "frequency_diagnostics_may_use_nominal_fs_labels": True,
        },
        "spokoiny_theory_alignment": {
            "irmf_domain": "normalized design domain t in [0,1]",
            "regular_design_interpretation": "t_i = i / n on [0,1]",
            "bandwidth_interpretation": "h_k is a normalized-domain bandwidth",
            "effective_local_sample_size": "n * h_k",
            "operator_scale_note": (
                "For fixed normalized h_k, increasing n increases the effective "
                "local sample size and changes discretization/local-estimation "
                "precision without changing the population-level normalized signal."
            ),
        },
        "sensitivity_modules": {
            "synthetic_discretization_sensitivity": {
                "status": "formal_sensitivity_required_not_yet_run",
                "priority": "high",
                "time_domain": "normalized_time",
                "n_values": [250, 500, 1000, 2000],
                "signal_definition": "same continuous normalized-time waveform on [0,1)",
                "scientific_question": (
                    "Are conclusions robust to sampling density and discrete "
                    "approximation of the same normalized-time signal?"
                ),
                "theoretical_rationale": (
                    "This aligns with the Spokoiny regular deterministic design: "
                    "changing n over [0,1] varies the regular-design density and "
                    "effective local sample size n*h_k while preserving the same "
                    "normalized-domain target function."
                ),
                "method_neutral_rationale": (
                    "For IRMF, n affects local robust-estimation precision under "
                    "fixed normalized bandwidths; for EMD-family methods, n affects "
                    "extrema localization, envelope interpolation, sifting behavior, "
                    "and small-scale IMF extraction."
                ),
                "recommended_scope": "representative_subset_before_any_full_extension",
                "recommended_outputs": [
                    "primary_metric_stability_by_n",
                    "dimension_rank_stability_by_n",
                    "method_ordering_stability_by_n",
                    "effective_component_count_by_n",
                    "splitting_merging_diagnostics_by_n",
                    "runtime_by_n",
                ],
                "must_not_be_described_as": "window_length_or_physical_duration_sensitivity",
            },
            "synthetic_physical_duration_sensitivity": {
                "status": "optional_secondary_not_core_not_implemented_not_run",
                "priority": "low_for_current_synthetic_benchmark",
                "time_domain": "physical_time",
                "fixed_fs": float(DEFAULT_FS),
                "candidate_window_durations_sec": [1.0, 2.0, 4.0],
                "candidate_n_values": [int(DEFAULT_FS * T) for T in (1.0, 2.0, 4.0)],
                "signal_definition_required": (
                    "synthetic generators must receive physical time t = arange(n)/fs, "
                    "so fixed-Hz components accumulate more cycles as T increases."
                ),
                "scientific_question": (
                    "Are conclusions robust to physical observation duration and "
                    "scale content when frequencies are held in physical units?"
                ),
                "not_required_for_core_benchmark_reason": (
                    "This is not the natural sensitivity implied by the normalized "
                    "Spokoiny formulation. After mapping a physical T-second signal "
                    "back to [0,1], fixed physical frequencies become normalized "
                    "frequencies proportional to T, so the experiment changes signal "
                    "complexity relative to the IRMF bandwidth scale as well as "
                    "observation horizon."
                ),
                "implementation_gate": (
                    "blocked until IRMF bandwidth scaling policy and frequency-label "
                    "semantics are frozen."
                ),
                "appropriate_triggers": [
                    "paper claims robustness across physical observation horizons",
                    "a reviewer requests window-length robustness",
                    "a dedicated low-frequency/baseline-wander synthetic study is added",
                ],
            },
            "real_ecg_window_duration_development": {
                "status": "formal_development_sensitivity_required_not_yet_run",
                "priority": "high",
                "time_domain": "physical_time",
                "native_sampling_rate_policy": "retain native ECG sampling rate unless separately justified",
                "candidate_window_durations_sec": [5.0, 10.0, 20.0],
                "scientific_question": (
                    "Which physical ECG window duration provides adequate R-peak counts, "
                    "stable validation endpoints, acceptable boundary behavior, and feasible runtime?"
                ),
                "rationale": (
                    "Unlike normalized synthetic signals, ECG is a physical-time "
                    "process. Window duration determines the number of annotated "
                    "beats, boundary fraction, baseline-wander observability, "
                    "endpoint discreteness, and runtime."
                ),
                "recommended_outputs": [
                    "r_peak_count_adequacy_by_T",
                    "endpoint_variance_by_T",
                    "component_stability_by_T",
                    "boundary_sensitivity_by_T",
                    "runtime_by_T",
                    "selected_main_window_duration_record",
                ],
            },
        },
        "irmf_bandwidth_scaling_options_for_physical_duration_sensitivity": {
            "option_A_normalized_fraction_of_window": {
                "description": (
                    "Keep h1 and h_min as fractions of the current analysis window. "
                    "As physical duration T increases, the physical smoothing span "
                    "in seconds increases proportionally."
                ),
                "current_params": {
                    "h1": float(GLOBAL_IRMF_PARAMS["h1"]),
                    "h_min": float(GLOBAL_IRMF_PARAMS["h_min"]),
                },
                "advantage": "closest to current normalized-time Spokoiny implementation",
                "risk": "changes the physical smoothing span across T",
            },
            "option_B_fixed_physical_duration": {
                "description": (
                    "Convert bandwidths so their physical duration in seconds remains "
                    "fixed across T, e.g. h_norm(T) = h_seconds / T."
                ),
                "advantage": "tests longer windows without changing nominal physical smoothing scale",
                "risk": (
                    "requires a new bandwidth interpretation and qualification before "
                    "comparison with the locked normalized-time benchmark"
                ),
            },
            "freeze_requirement": (
                "A physical-duration sensitivity module must choose one primary "
                "bandwidth-scaling policy before execution; reporting both policies "
                "is allowed only as a declared sensitivity analysis."
            ),
        },
        "frequency_diagnostic_semantics": {
            "risk": (
                "For normalized-time synthetic signals, frequency diagnostics that "
                "use nominal fs produce physical-Hz labels that may drift when n changes."
            ),
            "recommended_policy": (
                "Discretization sensitivity should report frequency diagnostics in "
                "normalized-frequency or index-spectrum terms unless a physical-time "
                "generator is used."
            ),
            "primary_metric_impact": (
                "This is mainly a secondary frequency-diagnostic interpretation risk; "
                "truth-based recovery and reconstruction primary endpoints do not "
                "require physical-Hz labels."
            ),
        },
        "paper_wording": (
            "Synthetic experiments were defined on a normalized time domain and "
            "therefore used sample size n as a discretization parameter rather than "
            "as a physical observation duration. Real-world ECG experiments, in "
            "contrast, retained the native sampling rate and defined analysis windows "
            "in physical time."
        ),
        "priority_decision": {
            "synthetic_required_sensitivity": "discretization_density_n",
            "synthetic_optional_sensitivity": "physical_observation_horizon_T",
            "real_data_required_sensitivity": "physical_window_duration_T",
            "summary": (
                "Synthetic tests discretization robustness; real data tests "
                "physical-window robustness. Synthetic physical-duration sensitivity "
                "is not required for the core IRMF benchmark."
            ),
        },
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(spec, output_root / "v531_time_axis_semantics_audit.json")
    return spec
