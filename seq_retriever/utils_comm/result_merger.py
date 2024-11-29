import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from icecream import ic
from loguru import logger
from pandas import DataFrame
from rapidfuzz.fuzz import partial_ratio, ratio

sys.path.append(os.path.abspath("."))
from seq_retriever.peptide_info.excluded_seqs.excluded_fake_seqs import (
    read_task_excluded_seqs,
)
from seq_retriever.peptide_info.fixed_peptides import fixed_pep_name_and_seqs
from seq_retriever.pmc.data_process import llama3_8_8B_result_file, pmc_input_file
from seq_retriever.utils_comm.aa_util import is_natural_seq
from seq_retriever.utils_comm.seq_parser import (
    EMPTY,
    INVALID_SEQS,
    TOO_LONG,
    UNAVAILABLE,
    calc_lower_triple_aa_num,
    calc_title_triple_aa_num,
    normalize_peptide_seq,
    pep_name_prefixes_to_filter,
    postprocess_seq_and_name,
    split_by_hyphen,
    strict_main_peptide_seq_pat,
    valid_peptide_seq_pat,
)
from seq_retriever.utils_comm.utils_retriever import parse_reply
from utils_comm.file_util import file_util

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEQ_PROCESSED_LABEL = "seq_postprocessed_label"
SEQ_NORMALIZED_LABEL = "seq_normalized_label"


class Merger:
    """ """

    def __init__(
        self,
        pmc_result_file=None,
        pmc_results=None,
        pmc_data_file=None,
        abstr_result_file=None,
        abstr_data_file=None,
        table_result_file=None,
        pickup=0,
        overwrite=1,
        task="anti_inflammatory_peptide",
        save_result=True,
        merged_result_file="",
        manually_annotated_file="",
        plot=0,
    ):
        if pmc_result_file is not None:
            self.pmc_result_file = Path(pmc_result_file)
            if pmc_results is not None:
                self.pmc_pred_results = pmc_results
            else:
                self.pmc_pred_results = file_util.read_json(pmc_result_file)
            logger.info(f"{pmc_result_file = }, {len(self.pmc_pred_results) = }")
            if pmc_data_file is not None:
                self.pmc_data = file_util.read_json(pmc_data_file)
                self.pmc_has_content = True
            else:
                self.pmc_data = [{} for _ in range(len(self.pmc_pred_results))]
                self.pmc_has_content = False
            assert len(self.pmc_data) == len(
                self.pmc_pred_results
            ), f"{len(self.pmc_data) = } != {len(self.pmc_pred_results) = }"

        if abstr_result_file is not None:
            self.abstr_result_file = Path(abstr_result_file)
            self.abstr_pred_results = file_util.read_json(abstr_result_file)
            logger.info(f"{abstr_result_file = }, {len(self.abstr_pred_results) = }")
            self.abstr_data = file_util.read_json(abstr_data_file)
            assert len(self.abstr_data) == len(
                self.abstr_pred_results
            ), f"{len(self.abstr_data) = } != {len(self.abstr_pred_results) = }"

        if table_result_file is not None:
            self.table_result_file = Path(table_result_file)
            self.table_pred_results = file_util.read_json(table_result_file)
            logger.info(f"{table_result_file = }, {len(self.table_pred_results) = }")

        self.pickup = pickup
        self.overwrite = overwrite
        self.excl_seqs, self.excl_prefixes = read_task_excluded_seqs(task)
        self.skipped_sections = ("merged_pred", "normalized_result")
        self.save_result = save_result
        self.merged_result_file = Path(merged_result_file)
        self.merged_result_file_raw = self.merged_result_file.with_stem(
            f"{self.merged_result_file.stem}_raw"
        )
        self.manually_annotated_file = manually_annotated_file
        self.plot = plot

    def is_better_name(self, name, normalized_seq: str, orig_seqs: list[str]):
        """A better name should have different informaton with normalized_seq and orig_seqs."""
        for seq in orig_seqs:
            if name in seq or seq in name:
                return False

        if name in normalized_seq or normalized_seq in name:
            return False
        return True

    def select_best_merged_result(self, row):
        """Called after raw_df.groupby("normalized_seq").agg(list).reset_index()

        Only row["normalized_seq"] is str and others are list.
        """
        orig_seq = row["orig_seq"]
        normalized_seq = row["normalized_seq"]
        names = row["name"]
        orig_name_seq_and_source = row["orig_name_seq_and_source"]
        best_name = names[0]
        for name in names:
            if not isinstance(name, str):
                continue
            if self.is_better_name(name, normalized_seq, orig_seq):
                best_name = name
                break
        row["name"] = best_name
        row["orig_name_seq_and_source"] = ";;".join(orig_name_seq_and_source)

        row["orig_seq"] = sorted(set(orig_seq))
        row["n_mod"] = sorted(set(s for s in row["n_mod"] if isinstance(s, str)))
        row["c_mod"] = sorted(set(s for s in row["c_mod"] if isinstance(s, str)))
        row["cyclic"] = sorted(set(row["cyclic"]))
        row["is_natural"] = is_natural_seq(normalized_seq)
        row["normalized_seq_len"] = len(normalized_seq)
        return row

    def overwrite_from_all_table_seqs(self, para_df):
        all_table_seqs_file = self.table_result_file.with_stem(
            f"{self.table_result_file.stem}_all_seqs"
        ).with_suffix(".csv")
        all_table_seqs = pd.read_csv(all_table_seqs_file)
        all_table_seqs = all_table_seqs.groupby("source").agg(list).reset_index()

        def check_name_unique(names):
            return len(set(names)) == len(names)

        unique_names_in_one_table = all_table_seqs[
            all_table_seqs["name"].map(check_name_unique)
        ]
        assert len(unique_names_in_one_table) == len(
            all_table_seqs
        ), f"{len(unique_names_in_one_table) = } != {len(all_table_seqs) = }"

        all_table_seqs_dict = defaultdict(dict)
        for _, row in all_table_seqs.iterrows():
            source = row["source"]
            names = row["name"]
            orig_seqs = row["orig_seq"]
            normalized_seqs = row["normalized_seq"]
            for name, orig_seq, normalized_seq in zip(
                names, orig_seqs, normalized_seqs
            ):
                all_table_seqs_dict[source][name] = [orig_seq, normalized_seq]

        def overwrite_from_table_seqs(row):
            source = row["source"]
            name = row["name"]
            table_source = source + "_table"
            item = all_table_seqs_dict.get(table_source, {}).get(name, None)
            if item:
                # 'Arginine-Glycine-Aspartic Acid', RGD
                orig_seq, normalized_seq = row["orig_seq"], row["normalized_seq"]
                # RGD, RGD
                table_orig_seq, table_normalized_seq = item

                if orig_seq != table_orig_seq and table_orig_seq != normalized_seq:
                    logger.info(
                        f"overwrite from table {source = }, {name = }, {orig_seq = }, "
                        f"{normalized_seq = }, table orig_seq/normalized_seq {item}"
                    )
                    row["orig_seq"] = table_orig_seq
                    row["normalized_seq"] = table_normalized_seq
            return row

        para_df = para_df.apply(overwrite_from_table_seqs, axis=1)
        return para_df

    def merge(self):
        """ merge the results of paragraph, abstract and table prediction."""
        para_df = self.merge_paragraph_pred()
        abstr_df = self.merge_abstract_pred()
        table_df = self.merge_table_pred()
        if self.manually_annotated_file:
            manually_annotated_df = pd.read_csv(self.manually_annotated_file)
            para_df = pd.concat([para_df, manually_annotated_df])
        para_df = self.overwrite_from_all_table_seqs(para_df)
        raw_df = pd.concat([para_df, abstr_df, table_df])
        logger.info(f"{len(raw_df) = }")
        raw_df = raw_df.drop_duplicates(subset=["source", "orig_seq"])

        unique_orig_seq_df = raw_df.groupby("orig_seq").agg(list).reset_index()
        logger.info(f"{len(unique_orig_seq_df) = }, {unique_orig_seq_df.columns = }")
        logger.info(f"{unique_orig_seq_df.iloc[0] = }")
        logger.info(f"{unique_orig_seq_df.iloc[1] = }")
        multi_names_df = unique_orig_seq_df[unique_orig_seq_df["name"].map(len) > 1]
        logger.info(f"{multi_names_df.iloc[0] = }")
        unique_orig_seq_df.to_csv(self.merged_result_file_raw, index=False, sep=",")

        raw_df["orig_name_seq_and_source"] = raw_df.apply(
            merge_orig_name_seq_and_source, axis=1
        )
        del raw_df["source"]
        normalized_seq_df = raw_df.groupby("normalized_seq").agg(list).reset_index()
        merged_df = normalized_seq_df.apply(self.select_best_merged_result, axis=1)
        del merged_df["orig_seq"]
        logger.info(f"{merged_df.iloc[0] = }")
        logger.info(f"{merged_df.iloc[1] = }")
        _natural_df = merged_df[
            merged_df["is_natural"] & (merged_df["normalized_seq_len"] > 2)
        ].copy()
        logger.info(f"All longths {len(_natural_df) = }")
        natural_df = _natural_df[_natural_df["normalized_seq_len"] <= 50]
        logger.info(f"{natural_df[['normalized_seq_len']].describe() = }")
        logger.info(f"{natural_df['normalized_seq_len'].value_counts() = }")
        natural_df.to_csv(self.merged_result_file, index=False, sep=",")

        too_long_seqs = _natural_df[_natural_df["normalized_seq_len"] > 50]
        too_long_seqs_file = self.merged_result_file.with_stem(
            f"{self.merged_result_file.stem}_too_long_seqs"
        ).with_suffix(".csv")
        if len(too_long_seqs) > 0:
            logger.info(f"{len(too_long_seqs) = }")
            too_long_seqs.to_csv(too_long_seqs_file, index=False, sep=",")

        if self.plot:
            plot_file = self.merged_result_file.with_suffix(".png")
            res = natural_df["normalized_seq_len"].plot.hist(bins=50).get_figure()
            res.savefig(plot_file)

    def merge_abstract_pred(self):
        """simplified merge_paragraph_pred, only one section: abstract, no labels."""
        logger.info("merge abstract results")
        valid_df = self.load_valid_seq_file(self.abstr_result_file)
        if valid_df is not None:
            return valid_df
        seqs = defaultdict(list)
        valid_normalized_seq_num = 0
        pep_name_num = 0
        pmid_abstr = {item['pmid']:item['abstract'] for item in self.abstr_data}
        for article_pred in self.abstr_pred_results:
            pmid = article_pred.get("pmid", "")
            # assert pmid == article_data["pmid"]
            if self.pickup and pmid != "24090768":
                continue
            logger.info(f'{pmid = }')
            merged_pred_raw = defaultdict(set)
            results = article_pred.get("abstract_pred", {})
            if  not results:
                # new version use "llm_replies" instead of "model_reply
                model_replies = article_pred.get("model_reply", {})
                results = {}
                for reply in model_replies.values():
                    result = parse_reply(reply)
                    for peptide_name, peptide_seqs in result.items():
                        former_set = results.get(peptide_name, set())
                        results[peptide_name] = former_set.union(peptide_seqs)
                for pep_name, pep_seqs in results.items():
                    results[pep_name] = sorted(pep_seqs)
            if self.pickup:
                ic(len(results))
                for pep_name, pep_seqs in results.items():
                    ic(pep_name, pep_seqs)
            # content = article_data.get("abstract", "")
            content = pmid_abstr[pmid]
            for pep_name, pep_seqs in results.items():
                for pep_seq in pep_seqs:
                    _pep_name, _pep_seq = postprocess_seq_and_name(
                        pep_name, pep_seq, content, self.excl_prefixes
                    )
                    if _pep_seq != EMPTY:
                        _pep_seq = self.re_process_abstract(
                            _pep_seq,
                            pep_seq,
                            results,
                            content,
                        )
                    if pep_seq in self.excl_seqs:
                        continue
                    merged_pred_raw[_pep_name].add(_pep_seq)
            self.post_merge(pmid, article_pred, merged_pred_raw)
            merged_orig = article_pred["merged_pred"]
            normalized_results = normalize_peptide_seq(merged_orig)
            article_pred["normalized_result"] = normalized_results
            pep_name_num += len(merged_orig)
            valid_normalized_seq_num = self.add_results(
                seqs, valid_normalized_seq_num, pmid, merged_orig, normalized_results
            )
            if self.pickup:
                logger.info(f"pickup {pmid = }")
                break
        natural_seq_num = sum(seqs["is_natural"])
        logger.info(
            f"abstract num = {len(self.abstr_data)}, {pep_name_num = }, {valid_normalized_seq_num = }, {natural_seq_num = }"
        )
        return self.save(seqs, self.abstr_result_file, self.abstr_pred_results)

    def save(self, seqs, result_file: Path, pred_results):
        valid_df = None
        if not self.pickup and self.save_result:
            merged_result_file = result_file.with_stem(f"{result_file.stem}_normalized")
            file_util.write_json(pred_results, merged_result_file)
            seqs_file = result_file.with_stem(f"{result_file.stem}_all_seqs")
            seqs_file = seqs_file.with_suffix(".csv")
            df = DataFrame(seqs)
            df.to_csv(seqs_file, index=False, sep=",")
            valid_df = df[~df["orig_seq"].isin(INVALID_SEQS)].copy()
            logger.info(f"valid orig seq num {len(valid_df)}")
            valid_seqs_file = seqs_file.with_stem(f"{result_file.stem}_valid_orig_seqs")
            valid_df.to_csv(valid_seqs_file, index=False, sep=',')

            valid_normalized_seqs_file = seqs_file.with_stem(
                f"{result_file.stem}_valid_normalized_seqs"
            )
            df['normalized_seq_len'] = df['normalized_seq'].map(len)
            valid_normalized_df = df[
                df["is_natural"]
                & (~df["normalized_seq"].isin(INVALID_SEQS))
                & (df["normalized_seq_len"] > 2)
                & (df["normalized_seq_len"] <= 50)
            ]
            logger.info(f"{len(valid_normalized_df) = }")
            valid_normalized_df.to_csv(valid_normalized_seqs_file, index=False, sep=",")
        return valid_df

    def merge_table_pred(self):
        """simplified merge_paragraph_pred, run as multi tables, no labels and no content.
        No content means only need pred result file, not need data file.
        """
        logger.info("merge table results")
        valid_df = self.load_valid_seq_file(self.table_result_file)
        if valid_df is not None:
            return valid_df
        seqs = defaultdict(list)
        valid_normalized_seq_num = 0
        pep_name_num = 0
        table_num = 0
        for table_result in self.table_pred_results:
            pmc = table_result['pmc']
            logger.info(f'table from {pmc = }')
            peptide_info = table_result['peptide_information']
            table_num += len(peptide_info)
            self.merge_sections_pred(
                pmc, peptide_info, {},
            )
            merged_orig = peptide_info["merged_pred"]
            normalized_results = normalize_peptide_seq(merged_orig)
            peptide_info['normalized_result'] = normalized_results
            pep_name_num += len(merged_orig)
            valid_normalized_seq_num = self.add_results(
                seqs, valid_normalized_seq_num, f"{pmc}_table", merged_orig, normalized_results
            )
            if self.pickup:
                logger.info(f"pickup {pmc = }")
                break
        natural_seq_num = sum(seqs["is_natural"])
        logger.info(
            f"table article num {len(self.table_pred_results)} {table_num = }, "
            f"{pep_name_num = } {valid_normalized_seq_num = } {natural_seq_num = }"
        )
        return self.save(seqs, self.table_result_file, self.table_pred_results)

    def load_valid_seq_file(self, result_file: Path):
        if not self.overwrite:
            valid_seqs_file = result_file.with_stem(
                f"{result_file.stem}_valid_orig_seqs"
            ).with_suffix(".csv")
            if valid_seqs_file.exists():
                logger.info(f"load {valid_seqs_file = }")
                valid_df = pd.read_csv(valid_seqs_file)
                natural_df = valid_df[~valid_df['normalized_seq'].isin(INVALID_SEQS)].copy()
                natural_df = natural_df[natural_df["is_natural"]]
                logger.info(f"{valid_seqs_file} exists, load it {len(valid_df) = }, {len(natural_df) = }.")
                return valid_df

    def merge_paragraph_pred(self):
        """supports to run test data with labels"""
        logger.info("merge paragraph results")
        valid_df = self.load_valid_seq_file(self.pmc_result_file)
        if valid_df is not None:
            return valid_df
        seqs = defaultdict(list)
        valid_normalized_seq_num = 0
        pep_name_num = 0
        pmc_paragraph = {item["pmc"]: item["paragraph"] for item in self.pmc_data if item}
        for article_pred in self.pmc_pred_results:
            pmc = article_pred.get("pmc", "")
            # if self.pmc_has_content:
            #    assert pmc == article_data["pmc"]
            logger.info(f'{pmc = }')
            paragraph_pred = article_pred["paragraph_pred"]
            paragraph = pmc_paragraph.get(pmc, {})
            seq_processed_labels = article_pred.get(SEQ_PROCESSED_LABEL, {})
            # PMC6349318 PMC7310322
            if self.pickup and pmc != "PMC3585349":
                continue
            self.merge_sections_pred(
                pmc, paragraph_pred, paragraph, seq_processed_labels
            )
            merged_orig = paragraph_pred["merged_pred"]
            normalized_results = normalize_peptide_seq(merged_orig)
            paragraph_pred["normalized_result"] = normalized_results
            labels = article_pred.get(SEQ_NORMALIZED_LABEL, {})

            # labels = {}  # disable normalized_labels check
            if labels:
                assert len(merged_orig) == len(
                    labels
                ), f"{len(merged_orig) = } {len(labels) = }, there may be unwanted normalized seqs in test data."
                label_seqs = list(labels.values())
                for i, (name, result) in enumerate(normalized_results.items()):
                    normalized_seq, n_mod, c_mod, cyclic, is_natural = result
                    seq_label = label_seqs[i]
                    assert seq_label == normalized_seq, (
                        f"{name = }, row {i+2} orig_seq = {merged_orig.get(name, '')}, "
                        f"{seq_label = } != {normalized_seq = }, {n_mod = } {c_mod = }"
                    )

            pep_name_num += len(merged_orig)
            valid_normalized_seq_num = self.add_results(
                seqs, valid_normalized_seq_num, pmc, merged_orig, normalized_results
            )
            if self.pickup:
                logger.info(f"pickup {pmc = }")
                break
        natural_seq_num = sum(seqs["is_natural"])
        logger.info(
            f"PMC acticle num = {len(self.pmc_data)}, {pep_name_num = }, "
            f"{valid_normalized_seq_num = }, {natural_seq_num = }"
        )
        return self.save(seqs, self.pmc_result_file, self.pmc_pred_results)

    def add_results(
        self,
        seqs,
        valid_normalized_seq_num,
        pmc_or_pmid,
        merged_orig,
        normalized_results,
    ):
        if merged_orig:
            for orig_seq, (pep_name, result) in zip(
                merged_orig.values(), normalized_results.items()
            ):
                normalized_seq, n_mod, c_mod, cyclic, is_natural = result
                if cyclic:
                    cyclic = True
                else:
                    cyclic = False
                seqs["name"].append(pep_name)
                seqs["orig_seq"].append(orig_seq)
                seqs["normalized_seq"].append(normalized_seq)
                seqs["source"].append(pmc_or_pmid)
                seqs["n_mod"].append(n_mod)
                seqs["c_mod"].append(c_mod)
                seqs["cyclic"].append(cyclic)
                seqs["is_natural"].append(is_natural)
                logger.info(
                    f"Final {pmc_or_pmid = } {pep_name = }\n{orig_seq = }"
                    f"\n{normalized_seq = }"
                )
                if normalized_seq not in INVALID_SEQS and orig_seq != EMPTY:
                    valid_normalized_seq_num += 1
        return valid_normalized_seq_num

    def merge_sections_pred(
        self, pmc, para_pred: dict, para: dict, seq_processed_labels=None
    ):
        """Merge the peptide seqs from different sections to one dict."""
        merged_pred_raw = defaultdict(set)
        row_num = 1
        for section_name, section_pred in para_pred.items():
            if section_name in self.skipped_sections:
                continue
            content = para.get(section_name, "")
            for pep_name, pep_seqs in section_pred.items():
                # compatible with old version output
                if isinstance(pep_seqs, str):
                    pep_seqs = [pep_seqs]
                for pep_seq in pep_seqs:
                    # if pep_name == "DRS-DA2N" and section_name == "Results":
                    #     ic(pep_name, pep_seq)
                    _pep_name, _pep_seq = postprocess_seq_and_name(
                        pep_name, pep_seq, content, self.excl_prefixes
                    )
                    # VLVLDTDYKK,VLVLDTDYKK
                    # IDALNENKVLVLDTDYKK,IDALNENKVLVLDTDYKK

                    # MFC,RCGNKRFRWHW -> RCGNKRFRWHW
                    # FC-PexS,MFC-PexS -> None
                    if _pep_seq != EMPTY and (
                        not strict_main_peptide_seq_pat.search(_pep_seq)
                        or len(_pep_seq) != len(_pep_name)
                    ):
                        _pep_seq = self.re_process_para(
                            section_name,
                            _pep_seq,
                            pep_seq,
                            para_pred,
                            para,
                        )
                    if seq_processed_labels:
                        row_num += 1
                        seq_true = seq_processed_labels[section_name][pep_name]
                        assert seq_true == _pep_seq, (
                            f"{row_num = }, {pmc = } {section_name = } {pep_name = } "
                            f"orig {pep_seq = }, {seq_true = } != {_pep_seq = }"
                        )
                    if _pep_seq in self.excl_seqs:
                        continue
                    merged_pred_raw[_pep_name].add(_pep_seq)
        self.post_merge(pmc, para_pred, merged_pred_raw)

    def post_merge(self, pmc_or_pmid, para_pred, merged_pred_raw):
        merged_pred = {}
        for pep_name, pep_seqs in merged_pred_raw.items():
            # merged_pred[pep_name] = sorted(pep_seqs)[0]  # debug to disable post merge
            merged_pred[pep_name] = select_possible_seq(pep_name, pep_seqs)
        merged_pred = drop_multi_names_for_one_valid_seq(merged_pred)
        merged_pred = disambiguate_same_seq_name_in_different_items(merged_pred)
        valid_seq_count = 0
        for pep_name, pep_seq in merged_pred.items():
            if pep_seq not in INVALID_SEQS:
                valid_seq_count += 1
        if valid_seq_count == 0 and len(merged_pred) >= 15:
            logger.warning(
                f"{pmc_or_pmid = } has as many as {len(merged_pred)} peptide names, but no one "
                f"valid seqs. Advice to manually check this paper"
            )

        para_pred["merged_pred"] = merged_pred
        if merged_pred:
            out_str = [f"{k}: {v}" for k, v in merged_pred.items()]
            out_str = "\n".join(out_str)
            logger.debug(f"{pmc_or_pmid = }, merged_orig\n{out_str}")

    def re_process_abstract(self, processed_seq, orig_seq, results, content):
        """similar as re_process_para, but simpler data structure, only 1 section: abstract"""
        if has_valid_triplets(processed_seq):
            return processed_seq
        processed_seq = self.re_process(processed_seq, orig_seq, results, content)
        return processed_seq

    def re_process_para(self, section_name, processed_seq, orig_seq, para_pred, para):
        """ """
        if has_valid_triplets(processed_seq):
            return processed_seq

        for section_name2, section_pred2 in para_pred.items():
            if section_name2 in self.skipped_sections:
                continue
            content = para.get(section_name, "")
            processed_seq = self.re_process(
                processed_seq, orig_seq, section_pred2, content
            )
            if processed_seq == EMPTY:
                return EMPTY
        return processed_seq

    def re_process(self, processed_seq, orig_seq, section_pred2, content):
        """
        case1:
        "DRS-DA2NE": [
            "DRS-DA2NE (sequence not provided)",
            "DRS-DA2NE"
        ]

        case2:
        MFC,RCGNKRFRWHW -> RCGNKRFRWHW
        MFC-PexS,MFC-PexS -> None
        "mromfunginc","MFC" -> None
        case2.1
        VLVLDTDYKK,VLVLDTDYKK,VLVLDTDYKK,PMC9586699,,,False,True
        IDALNENKVLVLDTDYKK,IDALNENKVLVLDTDYKK,IDALNENKVLVLDTDYKK,PMC9586699,,,False,False
        """
        for name2, pep_seqs2 in section_pred2.items():
            # compatible with old version output
            if isinstance(pep_seqs2, str):
                pep_seqs2 = [pep_seqs2]

            # case1, "DRS-DA2NE" in "DRS-DA2NE (sequence not provided)"
            # in case1 re run postprocess
            for seq2 in pep_seqs2:
                if seq2 != orig_seq and orig_seq in seq2:
                    _pep_name2, _pep_seq2 = postprocess_seq_and_name(
                        name2, seq2, content, self.excl_prefixes
                    )
                    if _pep_seq2 == EMPTY:
                        logger.info(f"{orig_seq} is None as {seq2} is None")
                        return EMPTY

            # case2, MFC in MFC-PexS, re run postprocess
            if name2 in processed_seq:
                _pep_name2, _pep_seq2 = postprocess_seq_and_name(
                    name2, seq2, content, self.excl_prefixes
                )
                # Must to add _pep_seq2 != processed_seq to avoid the same one
                if _pep_seq2 not in INVALID_SEQS and _pep_seq2 != processed_seq:
                    logger.info(
                        f"Other peptide name {name2} in this {processed_seq = }, and "
                        f"the seqs of the other pep name are valid {_pep_seq2 = } "
                        f"so treat this sequence with pep name as None"
                    )
                    return EMPTY
        return processed_seq


def merge_orig_name_seq_and_source(row):
    """merged result {name}::{orig_seq}::{source}"""
    name = row["name"]
    orig_seq = row["orig_seq"]
    source = row["source"]
    result = f"{name}::{orig_seq}::{source}"
    return result


def has_valid_triplets(processed_seq):
    triple_aa_upper_count = calc_title_triple_aa_num(processed_seq)
    triple_aa_lower_count = calc_lower_triple_aa_num(processed_seq)
    if triple_aa_upper_count > 2 or triple_aa_lower_count > 2:
        return True
    return False


def select_possible_seq(pep_name, pep_seqs: set[str]):
    """when pep_seqs length > 1, select the best possible seq.

    case1: select the last one
    "SPA4": "SPA4",
    "SPA4": ""FITC-SPA4"",
    "SPA4": "GDFRYSDGTPVNYTNWYRGE",
    """
    if len(pep_seqs) == 1:
        return pep_seqs.pop()

    pep_seqs.discard(EMPTY)
    if len(pep_seqs) == 1:
        return pep_seqs.pop()

    # that's len(pep_seqs) > 1 for the following
    pep_seqs.discard(UNAVAILABLE)
    if len(pep_seqs) == 1:
        return pep_seqs.pop()
    pep_seqs.discard(TOO_LONG)
    if len(pep_seqs) == 1:
        return pep_seqs.pop()

    # pep name is natural seq and len >=6, a strong signal to select the pep name as seq
    if len(pep_name) >= 6 and is_natural_seq(pep_name) and pep_name in pep_seqs:
        logger.info(f'{len(pep_name) = } >= 6 and is natural and in pep_seqs, so select it as seq.')
        return pep_name

    pep_seqs.discard(pep_name)
    if len(pep_seqs) == 1:
        return pep_seqs.pop()
    seqs = sorted(pep_seqs)
    if len(seqs) > 1:
        high_possible_seqs = []
        for seq in seqs:
            sub_words_in_seq = seq.split()
            if len(sub_words_in_seq) == 1:
                sub_words_in_seq = split_by_hyphen(seq)
            for sub_word in sub_words_in_seq:
                if sub_word in pep_name:
                    break
            else:
                high_possible_seqs.append(seq)
        if high_possible_seqs:
            logger.info(f"{pep_name = }, {high_possible_seqs = }")
            seqs = high_possible_seqs
    best_seq = seqs[0]
    if len(seqs) > 1:
        # fixed name and items, "MB2mP6": ["FEKEKL", "Myr", "", "", True],
        # pep_name = 'MB2mP6', seqs = ['HLPN', 'Myr-FEKEKL']
        for fixed_name, items in fixed_pep_name_and_seqs.items():
            fixed_seq = items[0]
            if fixed_name == pep_name:
                similar_scores = [partial_ratio(fixed_seq, seq) for seq in seqs]
                max_similar_score = max(similar_scores)
                if similar_scores.count(max_similar_score) > 1:
                    logger.info(f"More than 1 partial {max_similar_score = }")
                    similar_scores = [ratio(fixed_seq, seq) for seq in seqs]
                    max_similar_score = max(similar_scores)
                best_seq = seqs[similar_scores.index(max_similar_score)]
                logger.info(
                    f"Select {best_seq = } from {fixed_name = } {fixed_seq = }"
                )
                if max_similar_score < 60:
                    logger.warning(
                        f"{pep_name = }, {seqs = }, {fixed_name = }, {fixed_seq = }, "
                        f"{similar_scores = }, {best_seq = }. Similar score is too low!"
                    )
                break
        else:
            # PMC10654507 ['RFKAWRWAWRMKKLAAPS', 'RKRR', 'RMKKLAAPS']
            # generally filter steps
            # 1. first filter sub seq such as RMKKLAAPS in RFKAWRWAWRMKKLAAPS
            # 2. filter too short (<5) seqs
            _seqs = []
            for seq in seqs:
                is_sub_seq = False
                for _seq in seqs:
                    if seq != _seq and seq in _seq:
                        is_sub_seq = True
                        logger.info(f"{seq = } is sub seq of {_seq} and filter")
                        break
                if not is_sub_seq:
                    _seqs.append(seq)

            for seq in _seqs:
                if len(seq) > 4:
                    best_seq = seq
                    logger.warning(
                        f"Select pep seq long than 4 as {best_seq = } from {seqs}"
                    )
                    break
            else:
                logger.info(f"{best_seq = } is the first seq in {seqs}")
    return best_seq


def drop_multi_names_for_one_valid_seq(merged_pred: dict):
    seq2names = defaultdict(list)
    for name, seq in merged_pred.items():
        if seq not in INVALID_SEQS:
            seq2names[seq].append(name)
    seq2multi_names = {seq: names for seq, names in seq2names.items() if len(names) > 1}
    if seq2multi_names:
        logger.info(f"{seq2multi_names = }")
    seq2best_name = {}
    for seq, names in seq2multi_names.items():
        # delete the name which is the same as seq, if exists.
        if seq in names:
            names.remove(seq)
        # Just keep the first sorted name as the best name.
        best_name = names[0]
        for name in names:
            if name.startswith(pep_name_prefixes_to_filter):
                best_name = name
                break
        if len(names) > 1:
            logger.info(f"{seq} has multinames and choose {best_name = } from {names}.")
        seq2best_name[seq] = best_name
    new_merged_pred = {}
    for name, seq in merged_pred.items():
        if seq in seq2best_name:
            new_merged_pred[seq2best_name[seq]] = seq
        else:
            new_merged_pred[name] = seq
    return new_merged_pred


def disambiguate_same_seq_name_in_different_items(merged_orig: dict[str, str]):
    """After process in advance, the merged_orig dict has no duplicate name or no duplicate seq.

    when pep-name is in pred seqs, it may be wrong to treat seq as name, or treat name as seq.

    case1, its seq is None
    "PR-39": "RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP",
    "RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP": "None",

    case2, a fake one, the real example seq is CHR, explicit shortname and convert to None.
    one peptide seq is the name of another peptide, and this name is a shortname or
     partial name of full peptide name. So this peptide seq is wrong.
    CHF is the implicit shortname of fullname.
    "chromofungin": "CHF",
    "CHF": "CHGARILSILRHQNLLKELQDLAL",
    """
    logger.debug(f"Before disambiguation, {merged_orig = }")
    pred_seqs = list(merged_orig.values())
    seq_to_name = {seq: name for name, seq in merged_orig.items()}
    invalid_names_in_disambiguration = []
    for pep_name, pep_seq in merged_orig.items():
        if pep_name not in pred_seqs:
            continue
        # case1
        if pep_seq in INVALID_SEQS:
            logger.info(f"{pep_name = }, {pep_seq = } is dropped as the name is seq")
            invalid_names_in_disambiguration.append(pep_name)
            continue

        # case2 is duplicate as re-process in some logic, but keep.
        # As shortname or partial name, this pep name must be upper case.
        if pep_name.isupper() and pep_seq.isupper() and len(pep_seq) > len(pep_name):
            possible_full_pep_name = seq_to_name[pep_name]
            is_shortname_or_partial_name = False
            # is partial name
            if pep_name in possible_full_pep_name:
                is_shortname_or_partial_name = True
            else:
                # is shortname by very simple rules: all chars in pep_name are in possible_full_pep_name
                possible_full_pep_name_lower = possible_full_pep_name.lower()
                for char in pep_name:
                    if char.lower() not in possible_full_pep_name_lower:
                        break
                else:
                    is_shortname_or_partial_name = True
            if is_shortname_or_partial_name:
                _seq = merged_orig[possible_full_pep_name]
                logger.info(
                    f"{possible_full_pep_name = } seq {_seq} is converted to None, as "
                    f"{pep_name = } {pep_seq} is_shortname_or_partial_name"
                )
                merged_orig[possible_full_pep_name] = EMPTY
            continue

    # case3, PMC7310322, filter FITC-SPA4
    # "SPA4 peptide": "GDFRYSDGTPVNYTNWYRGE",
    # "FITC-SPA4": "FITC-SPA4",
    concise_names_seq = {
        name.replace(" peptide", ""): seq for name, seq in merged_orig.items()
    }
    for pep_name, pep_seq in merged_orig.items():
        if pep_name != pep_seq:
            continue
        for concise_name, seq in concise_names_seq.items():
            if (
                concise_name in pep_seq
                and len(concise_name) != len(seq)
                and concise_name != pep_name
                and f"{concise_name} peptide" != pep_name
                and valid_peptide_seq_pat.search(seq)
                and not has_valid_triplets(pep_name)
            ):
                logger.info(
                    f"{pep_name = }, {pep_seq = } is convert as None, because the other pep name inside the seq"
                )
                merged_orig[pep_name] = EMPTY
                break

    logger.debug(f"{invalid_names_in_disambiguration = }")
    for invalid_name in invalid_names_in_disambiguration:
        merged_orig.pop(invalid_name)
    return merged_orig


def check_merged_seq_num():
    """first version got 1541 seqs"""
    raw_data = file_util.read_json(llama3_8_8B_result_file)
    logger.info(f"{len(raw_data) = }")
    # valid_seq_num = 0
    # for k, v in only_seqs.items():
    #     for pep_name, pep_seq in v.items():
    #         if pep_seq not in INVALID_SEQS:
    #             valid_seq_num += 1
    # logger.info(f"{valid_seq_num = }")


def main():
    res_file = llama3_8_8B_result_file
    in_file = pmc_input_file
    merger = Merger(res_file, in_file, pickup=0)  # type: ignore
    merger.merge_paragraph_pred()


if __name__ == "__main__":
    main()
    # check_merged_seq_num()
    logger.info("end")