import logging
import random
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as lev_dist
from rdkit.ML.Cluster.Butina import ClusterData

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M",
    format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
)


def cluster_seqs_in_df_by_lev_dist(
    df: DataFrame,
    out_file=None,
    seq_col_name="Sequence",
    extra_filter_func=None,
    distThresh=8,
    random_seed=1,
    big_cluster_threshold=20,
    extra_sample_num_in_big_cluster=1,
):
    """
    https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html

    Args: extra_filter_func argument is a function that takes a row of input_df as input and returns True or False.
    """
    logger.info(f"cluster_seqs_in_df_by_lev_dist, len(input_df) {len(df)}")
    clusters = ClusterData(
        df[seq_col_name].tolist(), len(df), distThresh, distFunc=lev_dist
    )
    logger.info(f"len(clusters) {len(clusters)}")
    clusters_new = []
    if extra_filter_func is not None:
        for cluster in clusters:
            cluster_new = []
            for idx in cluster:
                if extra_filter_func(df.iloc[idx]):
                    cluster_new.append(idx)
            clusters_new.append(cluster_new)
    else:
        clusters_new = clusters

    idx = []
    random.seed(random_seed)
    for cluster in clusters_new:
        if len(cluster) > 0:
            idx.append(cluster[0])
        if len(cluster) > big_cluster_threshold:
            logger.info(f"big_cluster len(cluster) {len(cluster)}")
            cluster_pos_tmp = list(cluster)
            cluster_pos_tmp.remove(cluster[0])
            idx_plus = random.sample(cluster_pos_tmp, extra_sample_num_in_big_cluster)
            idx.extend(idx_plus)
    selected_df = df.iloc[idx].copy().reset_index(drop=True)
    logger.info(f"len(selected_df) {len(selected_df)}")
    if out_file:
        selected_df.to_csv(out_file, index=False, sep=",")

    return selected_df


def length_filter(seq, min_len=8, max_len=15):
    """ """
    if len(seq) < min_len or len(seq) > max_len:
        return False
    return True


if __name__ == "__main__":
    data_dir = Path(
        "/mnt/nas1/bio_drug_corpus/peptides/cyclic/GA_generation2/terminal_bond_cyclic-FGF5-prediction"
    )
    file = (
        data_dir
        / "terminal_bond_cyclic-FGF5-generated_terminal_bond_cyclic-FGF5_all_pos.csv"
    )
    df = pd.read_csv(file)
    logger.info(len(df))
    df = df[df["Sequence"].map(length_filter)]  # optional
    # df.to_csv(file.with_stem(f'{file.stem}_min_len8'), index=False, sep=',')
    cluster_seqs_in_df_by_lev_dist(
        df,
        distThresh=5,
        out_file=file.with_stem(f"{file.stem}_min_len8_distThresh5"),
        extra_sample_num_in_big_cluster=2,
    )
