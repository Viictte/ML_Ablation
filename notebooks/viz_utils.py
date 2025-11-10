"""Visualization helpers for SpikePointNet experiment notebooks."""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HistoryEntry = Mapping[str, float]
History = Sequence[HistoryEntry]
MetricMap = Mapping[str, Sequence[str]]


def _extract_series(history: History, key_candidates: Sequence[str]) -> Tuple[List[float], List[float]]:
    epochs: List[float] = []
    values: List[float] = []
    for index, record in enumerate(history):
        epoch = record.get('epoch', index + 1)
        value = None
        for key in key_candidates:
            if key in record and record[key] is not None:
                value = record[key]
                break
        if value is not None:
            epochs.append(epoch)
            values.append(value)
    return epochs, values


def plot_training_curves(
    histories: Mapping[str, History],
    metrics: MetricMap,
    *,
    title: str | None = None,
    max_cols: int = 2,
    figsize: Tuple[float, float] | None = None,
):
    """Plot per-epoch curves for each metric across one or more histories."""
    available_metrics: List[Tuple[str, Sequence[str]]] = []
    for metric_name, key_candidates in metrics.items():
        for history in histories.values():
            _, values = _extract_series(history, key_candidates)
            if values:
                available_metrics.append((metric_name, key_candidates))
                break
    if not available_metrics:
        raise ValueError("No metrics available for plotting.")

    cols = min(len(available_metrics), max_cols)
    rows = math.ceil(len(available_metrics) / cols)
    width = 6 * cols
    height = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (width, height), squeeze=False)

    for idx, (metric_name, key_candidates) in enumerate(available_metrics):
        ax = axes[idx // cols][idx % cols]
        for label, history in histories.items():
            epochs, values = _extract_series(history, key_candidates)
            if not values:
                continue
            ax.plot(epochs, values, marker='o', label=label)
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ax.lines:
            ax.legend()

    total_axes = rows * cols
    for idx in range(len(available_metrics), total_axes):
        ax = axes[idx // cols][idx % cols]
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig, axes


def summarize_histories(histories: Mapping[str, History], metrics: MetricMap):
    """Return first/last/best values for each metric in each history."""
    summary: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for label, history in histories.items():
        metric_summary: Dict[str, Dict[str, float | None]] = {}
        for metric_name, key_candidates in metrics.items():
            _, values = _extract_series(history, key_candidates)
            if values:
                metric_summary[metric_name] = {
                    'first': values[0],
                    'last': values[-1],
                    'best': max(values),
                }
            else:
                metric_summary[metric_name] = {'first': None, 'last': None, 'best': None}
        summary[label] = metric_summary
    return summary


def _format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return '-'
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def plot_metric_table(
    summary: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    *,
    title: str | None = None,
    value_fmt: str = '{:.4f}',
    include_first: bool = False,
    include_last: bool = True,
    include_best: bool = True,
    ax=None,
):
    """Render a table summarizing metric statistics for each configuration."""
    row_count = sum(len(metrics) for metrics in summary.values()) or 1
    if ax is None:
        fig_height = max(2.0, 0.5 * row_count)
        fig, ax = plt.subplots(figsize=(8, fig_height))
    else:
        fig = ax.figure

    columns = ['Variant', 'Metric']
    if include_first:
        columns.append('First')
    if include_last:
        columns.append('Last')
    if include_best:
        columns.append('Best')

    rows: List[List[str]] = []
    for label in sorted(summary.keys()):
        metrics = summary[label]
        for metric_name, stats in metrics.items():
            row = [label, metric_name]
            if include_first:
                row.append(_format_value(stats.get('first'), value_fmt))
            if include_last:
                row.append(_format_value(stats.get('last'), value_fmt))
            if include_best:
                row.append(_format_value(stats.get('best'), value_fmt))
            rows.append(row)

    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax


def plot_metric_bars(
    summary: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    metric_name: str,
    *,
    value_key: str = 'best',
    title: str | None = None,
    ylabel: str | None = None,
    ax=None,
):
    """Plot a bar chart highlighting a metric across configurations."""
    labels: List[str] = []
    values: List[float] = []
    for label in sorted(summary.keys()):
        metric_stats = summary[label].get(metric_name, {})
        value = metric_stats.get(value_key)
        if value is not None:
            labels.append(label)
            values.append(value)

    if not labels:
        raise ValueError(f'No values available for metric "{metric_name}".')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(3.0, len(labels) * 0.6)))
    else:
        fig = ax.figure

    bars = ax.bar(labels, values, color='#3776ab')
    ax.set_title(title or f'{metric_name} ({value_key.title()})')
    ax.set_ylabel(ylabel or metric_name)
    ax.set_ylim(0, max(values) * 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', ha='center', va='bottom')

    fig.tight_layout()
    return fig, ax
