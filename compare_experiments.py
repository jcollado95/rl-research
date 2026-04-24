#!/usr/bin/env python3
"""
Comparación de múltiples experimentos PIPO.

Genera tablas comparativas y gráficas superpuestas para analizar el efecto
de distintos hiperparámetros en los resultados de entrenamiento.

Uso:
    python compare_experiments.py experiments/exp_1/ experiments/exp_2/
    python compare_experiments.py experiments/2026-*/ --save
    python compare_experiments.py experiments/2026-*/ --latex
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


# ============================================================================
# Utilidades
# ============================================================================

COLORS_SEQUENCE = [
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#db2777", "#0891b2", "#65a30d",
]


def load_experiment_summary(experiment_dir):
    """Carga resumen + config + métricas de un directorio de experimento."""
    data = {"dir": experiment_dir, "name": os.path.basename(os.path.normpath(experiment_dir))}

    for fname, key in [("config.json", "config"), ("summary.json", "summary"), ("metrics.json", "metrics")]:
        fpath = os.path.join(experiment_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data[key] = json.load(f)
        else:
            data[key] = None

    return data


def ema_smooth(values, alpha=0.15):
    """Exponential moving average."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


# ============================================================================
# Comparison Table
# ============================================================================

def print_comparison_table(experiments):
    """Imprime tabla comparativa en formato markdown."""
    # Recopilar datos
    rows = []
    for exp in experiments:
        name = exp["name"]
        cfg = (exp.get("config") or {}).get("args", {})
        summ = exp.get("summary") or {}
        ini = summ.get("initial_metrics", {})
        fin = summ.get("final_metrics", {})

        rows.append({
            "Experiment": name,
            "BS": cfg.get("batch_size", "?"),
            "LR": cfg.get("lr", "?"),
            "Steps": cfg.get("num_steps", "?"),
            "KL β": cfg.get("kl_coeff", "?"),
            "LoRA r": cfg.get("lora_rank", "?"),
            "Init Cons.": f"{ini.get('consistency', 0):.4f}" if ini else "—",
            "Final Cons.": f"{fin.get('consistency', 0):.4f}" if fin else "—",
            "Best Cons.": f"{summ.get('best_consistency', 0):.4f}" if summ else "—",
            "Final ModalAcc": f"{fin.get('modal_accuracy', 0):.2%}" if fin else "—",
            "Final PerfAcc": f"{fin.get('perfect_accuracy', 0):.2%}" if fin else "—",
            "Wall Time": summ.get("wall_clock_human", "—"),
        })

    if not rows:
        print("⚠️  No hay datos para comparar.")
        return rows

    # Calcular anchos de columna
    headers = list(rows[0].keys())
    col_widths = {}
    for h in headers:
        col_widths[h] = max(len(h), max(len(str(r[h])) for r in rows))

    # Imprimir
    separator = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(f"\n  {separator}")
    print(f"  {' | '.join('-' * col_widths[h] for h in headers)}")
    for row in rows:
        line = " | ".join(f"{str(row[h]):<{col_widths[h]}}" for h in headers)
        print(f"  {line}")
    print()

    return rows


def export_latex_table(experiments, output_path=None):
    """Exporta la tabla comparativa en formato LaTeX."""
    rows = []
    for exp in experiments:
        cfg = (exp.get("config") or {}).get("args", {})
        summ = exp.get("summary") or {}
        ini = summ.get("initial_metrics", {})
        fin = summ.get("final_metrics", {})

        rows.append({
            "Experiment": exp["name"][:30],
            "BS": cfg.get("batch_size", "—"),
            "LR": cfg.get("lr", "—"),
            "Steps": cfg.get("num_steps", "—"),
            "Init Cons.": f"{ini.get('consistency', 0):.4f}" if ini else "—",
            "Final Cons.": f"{fin.get('consistency', 0):.4f}" if fin else "—",
            "Best Cons.": f"{summ.get('best_consistency', 0):.4f}" if summ else "—",
            "Modal Acc.": f"{fin.get('modal_accuracy', 0):.2%}" if fin else "—",
            "Perfect Acc.": f"{fin.get('perfect_accuracy', 0):.2%}" if fin else "—",
        })

    headers = list(rows[0].keys()) if rows else []
    n_cols = len(headers)

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        line = " & ".join(str(row[h]) for h in headers) + " \\\\"
        lines.append(line)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparison of experiment configurations and results.}")
    lines.append("\\label{tab:experiment_comparison}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"  📄 LaTeX table saved to: {output_path}")
    else:
        print("\n" + latex_str + "\n")

    return latex_str


# ============================================================================
# Overlay Plots
# ============================================================================

def plot_overlay(experiments, metric_key, ylabel, title, save_path=None, fmt="png", ylim=None):
    """Superpone la curva de una métrica de múltiples experimentos."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, exp in enumerate(experiments):
        metrics = exp.get("metrics")
        if not metrics or metric_key not in metrics[0]:
            continue

        steps = [m["step"] for m in metrics]
        values = [m[metric_key] for m in metrics]
        color = COLORS_SEQUENCE[i % len(COLORS_SEQUENCE)]
        label = exp["name"]

        # Raw (faint)
        ax.plot(steps, values, color=color, alpha=0.12, linewidth=0.7)

        # Smoothed
        smoothed = ema_smooth(values)
        ax.plot(steps, smoothed, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()

    if save_path:
        fname = f"compare_{metric_key}.{fmt}"
        out = os.path.join(save_path, fname)
        fig.savefig(out, format=fmt)
        print(f"  📊 Gráfica guardada: {out}")
        plt.close(fig)
    else:
        plt.show()


def plot_all_overlays(experiments, save_path=None, fmt="png"):
    """Genera gráficas superpuestas para todas las métricas principales."""
    metrics_to_plot = [
        ("consistency_score", "Score", "Consistency Score Comparison", (0, 1.05)),
        ("modal_accuracy", "Accuracy", "Modal Accuracy Comparison", (0, 1.05)),
        ("loss", "Loss", "Total Loss Comparison", None),
        ("mean_kl", "KL Divergence", "KL Divergence Comparison", None),
        ("grad_norm", "‖∇‖", "Gradient Norm Comparison", None),
    ]

    for metric_key, ylabel, title, ylim in metrics_to_plot:
        # Check if at least one experiment has this metric
        has_metric = any(
            exp.get("metrics") and len(exp["metrics"]) > 0 and metric_key in exp["metrics"][0]
            for exp in experiments
        )
        if has_metric:
            plot_overlay(experiments, metric_key, ylabel, title,
                         save_path=save_path, fmt=fmt, ylim=ylim)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar múltiples experimentos PIPO"
    )
    parser.add_argument(
        "experiment_dirs", nargs="+",
        help="Directorios de los experimentos a comparar"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guardar gráficas como imágenes"
    )
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="Formato de las gráficas (default: png)"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Exportar tabla comparativa en formato LaTeX"
    )
    parser.add_argument(
        "--latex-output", type=str, default=None,
        help="Archivo de salida para la tabla LaTeX (si no se especifica, se imprime en stdout)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configurar matplotlib
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "lines.linewidth": 1.5,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    if args.save:
        matplotlib.use("Agg")

    # Filtrar directorios válidos
    valid_dirs = [d for d in args.experiment_dirs if os.path.isdir(d)]
    if not valid_dirs:
        print("❌ No se encontraron directorios de experimentos válidos.")
        sys.exit(1)

    print(f"📂 Cargando {len(valid_dirs)} experimentos...")
    experiments = [load_experiment_summary(d) for d in valid_dirs]

    # Table
    print_comparison_table(experiments)

    # LaTeX
    if args.latex:
        export_latex_table(experiments, output_path=args.latex_output)

    # Overlay plots
    save_dir = None
    if args.save:
        # Save in parent directory of experiments
        parent = os.path.commonpath(valid_dirs) if len(valid_dirs) > 1 else os.path.dirname(valid_dirs[0])
        save_dir = os.path.join(parent, "comparison_plots")
        os.makedirs(save_dir, exist_ok=True)

    plot_all_overlays(experiments, save_path=save_dir, fmt=args.format)

    if save_dir:
        print(f"\n✅ Gráficas de comparación guardadas en: {save_dir}/")


if __name__ == "__main__":
    main()
