import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from utils_comm.log_util import logger
import numpy as np
from pathlib import Path
from scipy import stats
import math
from typing import List, Union


amino_acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "L",
    "M",
    "N",
    "P",
    "K",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def kde_plot_feature(
    dfs,
    feature_name="HydrophRatio",
    vertical_line_in_x=[],
    save_path=None,
    labels=["pos", "neg"],
    manual_xticks=None,
    title_prefix="",
    title="Kernel Distribution Estimation",
    label_fontsize=12,
    title_fontsize=15,
    xticks_fontsize=11,
    legend_fontsize=11,
):
    """
    manual_xticks: example, [-0.35, 0, 0.35, 0.7, 1.05, 1.4]
    """
    plt.figure(figsize=(6.229, 4.463))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    markersize = 4
    for i, df in enumerate(dfs):
        df[feature_name].plot.kde(
            label=labels[i],
            linewidth=1,
            # marker = "o", markersize=markersize, markevery=10,
        )
    for vertical_line_location_in_x in vertical_line_in_x:
        plt.axvline(
            x=vertical_line_location_in_x,
            c="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

    if manual_xticks:
        # keep the same format as plt.xticks()
        orig_xticks = [manual_xticks, 0]
    else:
        orig_xticks = plt.xticks()
    orig_locs = add_vertical_line_values_into_xticks(vertical_line_in_x, orig_xticks)
    plt.xticks(orig_locs, fontsize=xticks_fontsize)
    plt.yticks(fontsize=xticks_fontsize)

    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize=legend_fontsize)
    if title_prefix:
        title = title_prefix + " " + title.lower()
    plt.title(
        title, pad=10, loc="right", fontsize=title_fontsize, fontname="Times New Roman"
    )
    plt.xlabel(f"{feature_name} value", fontdict={"size": label_fontsize})
    plt.ylabel("Density", fontdict={"size": label_fontsize})
    sns.despine()
    plt.show()
    if save_path:
        logger.info(f"Save img file {save_path}")
        plt.savefig(save_path, bbox_inches="tight")


def kde_plot_feature_one_df(
    df,
    task_name,
    feature_name="len",
    vertical_line_in_x=[15],
    save_img_file=None,
    print_summary=False,
):
    plt.figure(figsize=(10, 6))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    markersize = 4
    df[feature_name].plot.kde(
        label=f"{task_name} {feature_name}",
        c="darkcyan",
        linewidth=1,
        marker="o",
        markersize=markersize,
        markevery=10,
    )

    for vertical_line_location_in_x in vertical_line_in_x:
        plt.axvline(
            x=vertical_line_location_in_x,
            c="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
    if vertical_line_in_x:
        orig_locs = add_vertical_line_values_into_xticks(
            vertical_line_in_x, plt.xticks()
        )
        plt.xticks(orig_locs)

    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    plt.xlabel(f"{feature_name} value")
    plt.ylabel(f"{feature_name} kde density")
    plt.show()
    if save_img_file:
        plt.savefig(save_img_file)
    if print_summary:
        logger.info(df[feature_name].describe())


def add_vertical_line_values_into_xticks(vertical_line_in_x, orig_xticks):
    orig_locs = orig_xticks[0]
    new_locs = []
    for vertical_locs in vertical_line_in_x:
        for locs in orig_locs:
            if locs <= vertical_locs:
                new_locs.append(locs)
            else:
                new_locs.append(vertical_locs)
                new_locs.append(locs)
        orig_locs = new_locs
        new_locs = []
    return orig_locs


def plot_proba_hist_kde(
    inputs,
    title,
    save_img_file=None,
    vertical_line_in_x=None,
    x_locs=None,
    enable_kde=False,
    bins: Union[str, int] = "auto",
    x_label="",
    change_small_to_zero=False,
):
    """ """
    normalized_inputs = inputs
    if change_small_to_zero:
        # A bug in sns.histplot, we have to if item < 1e-5: item = 0. This bug disappears.
        normalized_inputs = []
        for item in inputs:
            if item < 1e-5:
                item = 0
            normalized_inputs.append(item)

    plt.figure(figsize=(12, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    logger.info(f"Starts to plot hist, with kde {enable_kde}")
    sns.histplot(data=normalized_inputs, kde=enable_kde, bins=bins, discrete=True)
    if vertical_line_in_x:
        for vertical_line_location_in_x in vertical_line_in_x:
            plt.axvline(
                x=vertical_line_location_in_x,
                c="green",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )
    if x_locs:
        plt.xticks(x_locs)
    x_ticks_font = 20
    plt.xticks(fontsize=x_ticks_font)
    plt.title(title, fontsize=28)
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    if not x_label:
        x_label = "value"
    label_font = x_ticks_font + 2
    plt.xlabel(x_label, fontsize=label_font)
    plt.ylabel("count", fontsize=label_font)
    if save_img_file:
        plt.savefig(save_img_file)
    logger.info("Plots hist ends")


def hist_2_df_comparison(
    pos_df,
    neg_df,
    task_name,
    feature_name="",
    save_path=None,
    histtype="step",
    print_summary=False,
    bins=25,
):
    plt.figure(figsize=(16, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    if feature_name == "":
        feature_name = task_name
    neg_df[feature_name].hist(
        label=f"neg {task_name} {feature_name}", histtype=histtype, bins=bins
    )
    pos_df[feature_name].hist(
        label=f"pos {task_name} {feature_name}", histtype=histtype, bins=bins
    )
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    plt.ylabel("count")
    plt.xlabel("value")
    # plt.xticks([0, 0.38, 0.5, 1])
    plt.show()
    if save_path:
        plt.savefig(save_path)
    if print_summary:
        logger.info(f"neg_df {feature_name} summary describe()")
        logger.info(neg_df[feature_name].describe())
        logger.info(f"\npos_df {feature_name} summary describe()")
        logger.info(pos_df[feature_name].describe())


def hist_plot_feature_general(
    dfs,
    label_prefixes,
    feature,
    title,
    save_path=None,
    histtype="step",
    bins=15,
    print_summary=False,
    fontsize="medium",
):
    """bins should not be larger than the unqiue item num, otherwise there will be abnormal gap in plots."""
    assert len(dfs) == len(label_prefixes)
    plt.figure(figsize=(16, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    for df, label_prefix in zip(dfs, label_prefixes):
        df[feature].hist(
            label=f"{label_prefix} {feature}", histtype=histtype, bins=bins
        )
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize=fontsize)
    sns.despine()
    plt.tight_layout(pad=2)
    plt.ylabel("count")
    plt.xlabel("value")
    plt.title(title, pad=0, loc="center", fontsize=18, fontname="Times New Roman")
    # plt.xticks([0, 0.38, 0.5, 1])
    plt.show()
    if save_path:
        plt.savefig(save_path)
    if print_summary:
        for df, label_prefix in zip(dfs, label_prefixes):
            logger.info(f"{label_prefix} {feature} summary describe()")
            logger.info(df[feature].describe())


def calc_count_with_all_key(df, column, all_keys_in_fixed_order):
    """NB: all_keys_in_fixed_order, we need pre-run to initialize the dict to keep fixed sequence"""
    count = {}
    for v in all_keys_in_fixed_order:
        count[v] = 0
    for value in df[column]:
        count[round(value)] += 1
    return count


def get_each_aa_count(sequences, only_nature=True):
    aa_count = create_fixed_order_natural_aa_count()
    for seq in sequences:
        for aa in seq:
            if only_nature and aa not in amino_acids:
                continue
            aa_count[aa] += 1
    logger.info(f"len(aa_count) {len(aa_count)}")
    logger.info(f"aa_count {aa_count}")
    return aa_count


def create_fixed_order_natural_aa_count():
    """ """
    aa_count = defaultdict(int)
    for aa in amino_acids:
        aa_count[aa] = 0
    return aa_count


def get_terminal_aa_count(sequences, ternimal_type="C"):
    """ """
    aa_count = create_fixed_order_natural_aa_count()
    for seq in sequences:
        if ternimal_type == "C":
            aa_count[seq[-1]] += 1
        else:
            aa_count[seq[0]] += 1
    sorted_aac = sorted(aa_count.items(), key=lambda x: x[1])
    aa_count = {k: v for k, v in sorted_aac}
    logger.info(f"terminal {ternimal_type} aa_count {sorted_aac}")
    return aa_count


def plot_terminal_aa_count_ratio(
    title, pos_seq_lst, neg_seq_lst, ternimal_type, save_img_file=None
):
    """ """
    pos_aa_count = get_terminal_aa_count(pos_seq_lst, ternimal_type)
    neg_aa_count = get_terminal_aa_count(neg_seq_lst, ternimal_type)
    xlabel = "Amino acids"
    plot_fraction_contrastive_bars(
        (pos_aa_count, neg_aa_count), save_img_file, xlabel, title=title
    )


def plot_aa_count_ratio(title, pos_seq_lst, neg_seq_lst, save_img_file=None):
    pos_aa_count = get_each_aa_count(pos_seq_lst)
    neg_aa_count = get_each_aa_count(neg_seq_lst)
    xlabel = "Amino acids"
    plot_fraction_contrastive_bars(
        (pos_aa_count, neg_aa_count), save_img_file, xlabel, title=title
    )


def plot_aa_and_terminal_aa(pos_df, neg_df, plot_out_dir, task_name, postfix=""):
    """2 datasets, pos and neg."""
    plot_out_dir = Path(plot_out_dir)
    img_file = plot_out_dir / f"aa_composition{postfix}.png"
    plot_aa_count_ratio(task_name, pos_df["Sequence"], neg_df["Sequence"], img_file)

    img_file = plot_out_dir / f"terminal_N_aa_composition{postfix}.png"
    plot_terminal_aa_count_ratio(
        f"{task_name} terminal N", pos_df["Sequence"], neg_df["Sequence"], "N", img_file
    )

    img_file = plot_out_dir / f"terminal_C_aa_composition{postfix}.png"
    plot_terminal_aa_count_ratio(
        f"{task_name} terminal C", pos_df["Sequence"], neg_df["Sequence"], "C", img_file
    )


def plot_aa_count_ratio_single_group(task_name, seq_lst, save_img_file=None):
    """only one dataset"""
    aa_count = get_each_aa_count(seq_lst)
    xlabels = list(aa_count.keys())
    pos_values = np.array(list(aa_count.values()))
    pos_values_sum = pos_values.sum()
    pos_values = pos_values / pos_values_sum
    task_name = f"{task_name} aa count ratio in sequences"
    ylabel = "Count ratio"
    plot_bars(pos_values, xlabels, ylabel, task_name, save_img_file)


def plot_terminal_aa_count_ratio_single_group(
    title, seq_lst, ternimal_type, save_img_file=None
):
    """ """
    aa_count = get_terminal_aa_count(seq_lst, ternimal_type)
    xlabels = list(aa_count.keys())
    pos_values = np.array(list(aa_count.values()))
    pos_values_sum = pos_values.sum()
    pos_values = pos_values / pos_values_sum
    title = f"{title} terminal {ternimal_type} aa count ratio in sequences"
    ylabel = "Count ratio"
    plot_bars(pos_values, xlabels, ylabel, title, save_img_file)


def plot_aa_and_terminal_aa_single_group(seqs, plot_out_dir, task_name, postfix=""):
    """ """
    plot_out_dir = Path(plot_out_dir)
    img_file = plot_out_dir / f"{task_name}_aac{postfix}.png"
    plot_aa_count_ratio_single_group(task_name, seqs, img_file)

    img_file = plot_out_dir / f"{task_name}_terminal_N_aa_composition{postfix}.png"
    plot_terminal_aa_count_ratio_single_group(task_name, seqs, "N", img_file)

    img_file = plot_out_dir / f"{task_name}_terminal_C_aa_composition{postfix}.png"
    plot_terminal_aa_count_ratio_single_group(task_name, seqs, "C", img_file)


def plot_aa_count_ratio_contrastive_bars_df(
    dfs, title, column, save_img_file, labels=None
):
    """sort aac keys by the first df aa fraction values"""
    aa_counts = []
    for df in dfs:
        aa_count = get_each_aa_count(df[column])
        aa_counts.append(aa_count)
    xlabel = "Amino acids"

    plot_fraction_contrastive_bars(
        aa_counts,
        save_img_file,
        xlabel,
        title=title,
        labels=labels,
        sort_by_frac_values=True,
    )


def get_fraction_values(input_account: dict, xlabels):
    """ """
    values = np.array([input_account.get(k, 0) for k in xlabels])
    values_sum = values.sum()
    fraction_values = values / values_sum
    return fraction_values


def plot_fraction_contrastive_bars(
    counts,
    save_img_file,
    xlabel,
    title=None,
    labels=None,
    selected_xticks=None,
    sort_by_frac_values=False,
):
    ylabel = "Fraction"
    # Not shown all xlabels, but partial, selectes the most frequent partial.
    if selected_xticks:
        frac_values = [get_fraction_values(count, selected_xticks) for count in counts]
        xticks = selected_xticks
    else:
        xticks = list(counts[0].keys())
        frac_values = [get_fraction_values(aa_count, xticks) for aa_count in counts]

    if sort_by_frac_values:
        sorted_indexes = np.argsort(frac_values[0])
        xticks = [xticks[i] for i in sorted_indexes]
        _frac_values = []
        for frac_value in frac_values:
            _frac_values.append([frac_value[i] for i in sorted_indexes])
        frac_values = _frac_values

    plot_contrastive_bars(
        frac_values, xticks, xlabel, ylabel, save_img_file, title, labels
    )


def plot_fraction_hist_in_dfs(
    dfs,
    column_name,
    labels,
    save_path,
    title="",
    xlabel="",
    func=None,
    selected_xlabels=None,
    seq_col="Sequence",
    label_with_num=True,
):
    """ """
    assert len(labels) == len(dfs)
    for df in dfs:
        if column_name not in df:
            df[column_name] = df[seq_col].map(func)
    min_v = math.inf
    max_v = -math.inf
    for df in dfs:
        min_v = min(min(df[column_name]), min_v)
        max_v = max(max(df[column_name]), max_v)
    min_v = round(min_v)
    max_v = round(max_v)
    all_keys_in_order = list(range(min_v, max_v + 1))
    logger.info("min_v %s, max_v %s", min_v, max_v)
    counts = [calc_count_with_all_key(df, column_name, all_keys_in_order) for df in dfs]
    column_name = column_name[0].upper() + column_name[1:]
    if not title:
        title = column_name
    if not xlabel:
        xlabel = column_name
    if label_with_num:
        _labels = []
        for label, num in zip(labels, dfs):
            _labels.append(f'{label}, num {num}')
        labels = _labels
    plot_fraction_contrastive_bars(
        counts,
        save_path,
        xlabel,
        title=title,
        labels=labels,
        selected_xticks=selected_xlabels,
    )


def plot_contrastive_bars(
    all_values,
    xticks,
    xlabel,
    ylabel,
    save_img_file,
    title=None,
    labels=None,
    x_fontzie=11,
    figsize=(9, 6),
    rotation=0,
):
    """ """
    plt.figure(figsize=figsize)
    sns.set_style(style="ticks")
    sns.set_context("talk")
    n_bars = len(all_values)
    x = np.arange(len(xticks))

    if not labels:
        labels = ("Positive", "Negative")
    total_width = 0.72
    bar_width = total_width / n_bars
    for i, values in enumerate(all_values):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        plt.bar(x + x_offset, values, bar_width, label=labels[i])
    plt.xlabel(xlabel, fontdict={"size": 12})
    plt.ylabel(ylabel, fontdict={"size": 12})
    if title:
        plt.title(title, pad=10, loc="center", fontsize=15, fontname="Times New Roman")
    plt.xticks(x, xticks, fontsize=x_fontzie, rotation=rotation, ha="right")
    plt.yticks(fontsize=x_fontzie)
    plt.legend(fontsize=12)
    sns.despine()
    ## there is abnormal display error when the value is < 1, skip to add text
    # for x1, y1 in enumerate(pos_values):
    #     plt.text(x1 - width/2, y1+1, round(y1, 1), fontsize=8, ha='center')
    # for x2, y2 in enumerate(neg_values):
    #     plt.text(x2 + width/2, y2+1, round(y1, 2), fontsize=8, ha='center')
    plt.show()
    if save_img_file:
        logger.info(f"Save img file {save_img_file}")
        plt.savefig(save_img_file, bbox_inches="tight")


def plot_bars(
    values, xlabels, ylabel, title, save_img_file, x_fontzie=None, figsize=(13, 9)
):
    """ """
    plt.figure(figsize=figsize)
    sns.set_style(style="ticks")
    sns.set_context("talk")
    x = np.arange(len(xlabels))
    width = 0.35
    plt.bar(x - width / 2, values, width, label="values")
    plt.ylabel(ylabel)
    plt.title(title, pad=10)
    if x_fontzie:
        plt.xticks(x, xlabels, fontsize=x_fontzie)
    else:
        plt.xticks(x, xlabels)
    plt.legend()
    ## there is abnormal display error when the value is < 1, skip to add text
    # for x1, y1 in enumerate(pos_values):
    #     plt.text(x1 - width/2, y1+1, round(y1, 1), fontsize=8, ha='center')
    # for x2, y2 in enumerate(neg_values):
    #     plt.text(x2 + width/2, y2+1, round(y1, 2), fontsize=8, ha='center')
    plt.show()
    if save_img_file:
        plt.savefig(save_img_file)


def calc_spearmanr(x, y, notes=""):
    """ """
    res = stats.spearmanr(x, y)
    logger.info(f"{notes} spearmanr: {res}")
    if hasattr(res, "correlation"):
        spearman_ratio = float(res.correlation)  # type: ignore
    else:
        assert hasattr(res, "statistic")
        spearman_ratio = float(res.statistic)  # type: ignore
    if hasattr(res, "pvalue"):
        pvalue = float(res.pvalue)  # type: ignore
        logger.info(f"{notes} spearmanr pvalue: {pvalue}")
    logger.info(f"{notes} spearman_ratio: {spearman_ratio}")
    return spearman_ratio


def plot_scatter(
    x,
    y,
    save_file,
    title="",
    fontsize=16,
    plot_diagonal=True,
    value_name="value",
    xlabel="",
    ylabel="",
    calc_spearman=True,
):
    """
    plot performance of regression model against validation dataset, x can be experimental and y predicted values.
    title with f'spearman correlation ratio {spearmanr}'
    """
    plt.figure(figsize=(16, 10))
    plt.scatter(x, y)
    plt.grid(visible=True)
    if not xlabel:
        xlabel = f"Real {value_name}"
    if not ylabel:
        ylabel = f"Predicted {value_name}"
    plt.xlabel(xlabel, fontdict={"size": fontsize})
    plt.ylabel(ylabel, fontdict={"size": fontsize})
    spearman_ratio = None
    if calc_spearman:
        spearman_ratio = calc_spearmanr(x, y)
        if not title:
            title = f"spearman correlation ratio {spearman_ratio:.4f}"
        else:
            title = title + f", spearman correlation ratio {spearman_ratio:.4f}"
        logger.info(title)
    if title:
        plt.title(title, fontdict={"size": fontsize + 2})
    if plot_diagonal:
        x = plt.xticks()[0]
        plt.plot(x, x, c="green", linestyle="--", linewidth=1, alpha=0.5)
    plt.tick_params(labelsize=14)
    plt.savefig(save_file)
    return spearman_ratio


def plot_lines(ranking_scores, recalls, f1s, precisions, plot_dir):
    """An example to plot multi plines"""
    title = "recall_precision_with_different_ranking scores"
    fontsize = 16
    plt.figure(figsize=(16, 10))
    plt.plot(ranking_scores, recalls, c="blue", label="recall")
    plt.plot(ranking_scores, f1s, c="cyan", label="f1")
    plt.plot(ranking_scores, precisions, c="green", label="precision")
    plt.grid(visible=True)
    plt.title(title, fontdict={"size": fontsize + 2})
    plt.legend(fontsize=fontsize)
    plt.xlabel("ranking scores", fontsize=fontsize)
    plt.ylabel("recall precision ratio", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_dir / f"{title}.png")


def plot_violin(
    x_column,
    y_column,
    data,
    img_file,
    title_fontsize=15,
    label_fontsize=12,
    xticks_fontsize=11,
):
    """ """
    init_plot_style()
    sns.violinplot(x=x_column, y=y_column, data=data, inner="quartile")
    plt.title(y_column, loc="center", fontsize=title_fontsize)
    sns.despine()
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=xticks_fontsize)

    plt.xlabel("")
    plt.ylabel(y_column, fontdict={"size": label_fontsize})
    plt.savefig(img_file, bbox_inches="tight")


def init_plot_style(style="ticks", context="talk"):
    """ """
    sns.set_style(style=style)
    sns.set_context(context)


if __name__ == "__main__":
    pass
