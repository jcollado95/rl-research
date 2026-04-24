#!/usr/bin/env python3
"""
Visualización de resultados de experimentos PIPO.

Genera gráficas de calidad publicación a partir de los datos de un directorio
de experimento (metrics.json, summary.json, eval/*.json).

Uso:
    python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/
    python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/ --save
    python visualize_results.py experiments/2026-04-23_11-08_bs4_lr5e-06/ --save --format pdf
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================================
# Configuración global de estilo
# ============================================================================

COLORS = {
    "primary":    "#2563eb",   # blue-600
    "secondary":  "#dc2626",   # red-600
    "accent":     "#059669",   # emerald-600
    "warning":    "#d97706",   # amber-600
    "purple":     "#7c3aed",   # violet-600
    "gray":       "#6b7280",   # gray-500
    "light_gray": "#e5e7eb",   # gray-200
    "raw_alpha":  0.15,
}

METRIC_CONFIG = {
    "loss": {
        "label": "Total Loss",
        "color": COLORS["primary"],
        "ylabel": "Loss",
    },
    "reinforce_loss": {
        "label": "REINFORCE Loss",
        "color": COLORS["secondary"],
        "ylabel": "Loss",
    },
    "consistency_score": {
        "label": "Consistency Score (train batch)",
        "color": COLORS["accent"],
        "ylabel": "Score",
        "ylim": (0, 1.05),
    },
    "modal_accuracy": {
        "label": "Modal Accuracy (train batch)",
        "color": COLORS["warning"],
        "ylabel": "Accuracy",
        "ylim": (0, 1.05),
    },
    "mean_kl": {
        "label": "Mean KL Divergence",
        "color": COLORS["purple"],
        "ylabel": "KL",
    },
    "grad_norm": {
        "label": "Gradient Norm",
        "color": COLORS["gray"],
        "ylabel": "‖∇‖",
    },
}


def setup_style():
    """Configura matplotlib para gráficas de publicación."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def ema_smooth(values, alpha=0.1):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


def load_experiment(experiment_dir):
    """Carga todos los datos de un directorio de experimento."""
    data = {"dir": experiment_dir}

    # Métricas de entrenamiento
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    else:
        print(f"⚠️  No se encontró {metrics_path}")
        data["metrics"] = []

    # Resumen
    summary_path = os.path.join(experiment_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = None

    # Config
    config_path = os.path.join(experiment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            data["config"] = json.load(f)
    else:
        data["config"] = None

    # Evaluaciones
    eval_dir = os.path.join(experiment_dir, "eval")
    data["evals"] = {}
    if os.path.isdir(eval_dir):
        for fname in os.listdir(eval_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(eval_dir, fname)
                with open(fpath) as f:
                    data["evals"][fname] = json.load(f)

    # Also check for legacy eval files in the experiment dir root
    for fname in os.listdir(experiment_dir):
        if fname.startswith("eval_results") and fname.endswith(".json"):
            fpath = os.path.join(experiment_dir, fname)
            with open(fpath) as f:
                data["evals"][fname] = json.load(f)

    return data


# ============================================================================
# Training Curves
# ============================================================================

def plot_training_curves(data, save_path=None, fmt="png"):
    """Genera gráficas de curvas de entrenamiento (multi-panel)."""
    metrics = data["metrics"]
    if not metrics:
        print("⚠️  Sin datos de entrenamiento para graficar.")
        return

    steps = [m["step"] for m in metrics]

    # Detectar qué métricas están disponibles
    available = [k for k in METRIC_CONFIG if k in metrics[0]]
    n_plots = len(available)

    if n_plots == 0:
        print("⚠️  No se encontraron métricas reconocidas en metrics.json")
        return

    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(12, 3.2 * n_plots),
        sharex=True,
    )
    if n_plots == 1:
        axes = [axes]

    # Detectar evaluation steps para líneas verticales
    eval_steps = [m["step"] for m in metrics if "val_consistency" in m]

    for ax, metric_key in zip(axes, available):
        cfg = METRIC_CONFIG[metric_key]
        values = [m[metric_key] for m in metrics]

        # Raw data (faint)
        ax.plot(
            steps, values,
            color=cfg["color"],
            alpha=COLORS["raw_alpha"],
            linewidth=0.8,
        )

        # Smoothed curve
        smoothed = ema_smooth(values, alpha=0.15)
        ax.plot(
            steps, smoothed,
            color=cfg["color"],
            linewidth=2.0,
            label=f"{cfg['label']} (EMA)",
        )

        # Eval checkpoint lines
        for es in eval_steps:
            ax.axvline(x=es, color=COLORS["light_gray"], linestyle="--", linewidth=0.8)

        # Validation points if available
        if metric_key == "consistency_score":
            val_steps = [m["step"] for m in metrics if "val_consistency" in m]
            val_values = [m["val_consistency"] for m in metrics if "val_consistency" in m]
            if val_values:
                ax.scatter(val_steps, val_values, color=COLORS["secondary"],
                           s=40, zorder=5, label="Val consistency", marker="D")

        if metric_key == "modal_accuracy":
            val_steps = [m["step"] for m in metrics if "val_modal_accuracy" in m]
            val_values = [m["val_modal_accuracy"] for m in metrics if "val_modal_accuracy" in m]
            if val_values:
                ax.scatter(val_steps, val_values, color=COLORS["secondary"],
                           s=40, zorder=5, label="Val modal acc", marker="D")

            # Also overlay perfect accuracy on this panel
            pa_steps = [m["step"] for m in metrics if "val_perfect_accuracy" in m]
            pa_values = [m["val_perfect_accuracy"] for m in metrics if "val_perfect_accuracy" in m]
            if pa_values:
                ax.scatter(pa_steps, pa_values, color=COLORS["purple"],
                           s=40, zorder=5, label="Val perfect acc", marker="s")

        ax.set_ylabel(cfg["ylabel"])
        if "ylim" in cfg:
            ax.set_ylim(cfg["ylim"])
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_title(cfg["label"], fontweight="bold", loc="left")

    axes[-1].set_xlabel("Training Step")

    # Global title
    exp_name = os.path.basename(os.path.normpath(data["dir"]))
    fig.suptitle(f"Training Curves — {exp_name}", fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        out = os.path.join(save_path, f"training_curves.{fmt}")
        fig.savefig(out, format=fmt)
        print(f"  📊 Gráfica guardada: {out}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# Evaluation Comparison
# ============================================================================

def plot_eval_comparison(data, save_path=None, fmt="png"):
    """Genera gráficas comparando evaluaciones (base vs entrenado)."""
    evals = data["evals"]
    if not evals:
        print("⚠️  Sin datos de evaluación para graficar.")
        return

    # Recopilar métricas de cada evaluación
    eval_summaries = {}
    for fname, eval_data in evals.items():
        if "metrics" in eval_data:
            label = fname.replace("eval_results_", "").replace("eval_", "").replace(".json", "")
            if not label:
                label = "default"
            eval_summaries[label] = eval_data["metrics"]

    if not eval_summaries:
        print("⚠️  No se encontraron métricas en los archivos de evaluación.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Bar chart de métricas principales ---
    ax1 = axes[0]
    metric_names = ["mean_consistency", "modal_accuracy", "perfect_accuracy"]
    metric_labels = ["Consistency", "Modal Acc.", "Perfect Acc."]
    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(eval_summaries), 1)
    bar_colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["purple"]]

    for i, (label, metrics) in enumerate(eval_summaries.items()):
        values = [metrics.get(m, 0) or 0 for m in metric_names]
        offset = (i - (len(eval_summaries) - 1) / 2) * width
        bars = ax1.bar(x + offset, values, width * 0.9,
                       label=label, color=bar_colors[i % len(bar_colors)],
                       edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2%}", ha="center", va="bottom", fontsize=8,
                     fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Evaluation Metrics", fontweight="bold", loc="left")
    ax1.legend()

    # --- Panel 2: Positional bias ---
    ax2 = axes[1]
    letters = ["A", "B", "C"]
    ideal = 1.0 / 3  # ~33.3%

    for i, (label, metrics) in enumerate(eval_summaries.items()):
        bias = metrics.get("positional_bias", {})
        values = [bias.get(l, 0) for l in letters]
        offset = (i - (len(eval_summaries) - 1) / 2) * width
        ax2.bar(np.arange(len(letters)) + offset, values, width * 0.9,
                label=label, color=bar_colors[i % len(bar_colors)],
                edgecolor="white", linewidth=0.5)

    ax2.axhline(y=ideal, color=COLORS["secondary"], linestyle="--",
                linewidth=1.5, label=f"Ideal ({ideal:.1%})")
    ax2.set_xticks(np.arange(len(letters)))
    ax2.set_xticklabels(letters, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_ylabel("Selection Rate")
    ax2.set_title("Positional Bias", fontweight="bold", loc="left")
    ax2.legend()

    exp_name = os.path.basename(os.path.normpath(data["dir"]))
    fig.suptitle(f"Evaluation Results — {exp_name}", fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        out = os.path.join(save_path, f"eval_comparison.{fmt}")
        fig.savefig(out, format=fmt)
        print(f"  📊 Gráfica guardada: {out}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# Summary Table
# ============================================================================

def print_summary(data):
    """Imprime una tabla-resumen del experimento."""
    summary = data.get("summary")
    config = data.get("config")

    exp_name = os.path.basename(os.path.normpath(data["dir"]))

    print(f"\n{'='*65}")
    print(f"  📋 Experiment Summary: {exp_name}")
    print(f"{'='*65}")

    if config:
        args = config.get("args", {})
        print(f"  Timestamp:      {config.get('timestamp', 'N/A')}")
        print(f"  Git hash:       {config.get('git_hash', 'N/A')}")
        print(f"  Model:          {args.get('model_name', 'N/A')}")
        print(f"  Batch size:     {args.get('batch_size', 'N/A')}")
        print(f"  Learning rate:  {args.get('lr', 'N/A')}")
        print(f"  Num steps:      {args.get('num_steps', 'N/A')}")
        print(f"  KL coeff:       {args.get('kl_coeff', 'N/A')}")
        print(f"  LoRA rank:      {args.get('lora_rank', 'N/A')}")

    if summary:
        print(f"\n  Wall clock:     {summary.get('wall_clock_human', 'N/A')}")
        print(f"  Best consistency: {summary.get('best_consistency', 'N/A'):.4f}")

        ini = summary.get("initial_metrics", {})
        fin = summary.get("final_metrics", {})
        print(f"\n  {'Metric':<22} {'Initial':>10} {'Final':>10} {'Δ':>10}")
        print(f"  {'─'*54}")
        for key in ["consistency", "modal_accuracy", "overall_accuracy", "perfect_accuracy"]:
            iv = ini.get(key, 0)
            fv = fin.get(key, 0)
            delta = fv - iv
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<22} {iv:>10.4f} {fv:>10.4f} {sign}{delta:>9.4f}")

    print(f"{'='*65}\n")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizar resultados de un experimento PIPO"
    )
    parser.add_argument(
        "experiment_dir",
        help="Directorio del experimento a visualizar"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guardar gráficas como imágenes en lugar de mostrarlas"
    )
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="Formato de salida para las gráficas (default: png)"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="No imprimir tabla-resumen en la terminal"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    if not os.path.isdir(args.experiment_dir):
        print(f"❌ No existe el directorio: {args.experiment_dir}")
        sys.exit(1)

    print(f"📂 Cargando experimento: {args.experiment_dir}")
    data = load_experiment(args.experiment_dir)

    # Summary
    if not args.no_summary:
        print_summary(data)

    # Plots
    save_dir = os.path.join(args.experiment_dir, "plots") if args.save else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use non-interactive backend when saving
    if args.save:
        matplotlib.use("Agg")

    plot_training_curves(data, save_path=save_dir, fmt=args.format)
    plot_eval_comparison(data, save_path=save_dir, fmt=args.format)

    if args.save:
        print(f"\n✅ Gráficas guardadas en: {save_dir}/")


if __name__ == "__main__":
    main()
