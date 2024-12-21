#!/usr/bin/env python
# coding=utf-8
import torch
import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

# ref: https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/data_loader.py

def stastic_ts(data):
    dl = data.sum(dim=1).cpu().numpy()
    # 计算平均值
    mean_value = np.mean(dl)
    # 计算方差（无偏估计，即样本方差）
    variance_value = np.var(dl, ddof=1)
    # 计算中位数
    median_value = np.median(dl)
    # 计算众数
    counter = Counter(dl)
    most_common_element, most_common_count = counter.most_common(1)[0]
    mode_value = most_common_element
    # 如果数组中有多个众数（出现次数相同且最多），则返回第一个找到的
    # 注意：这个函数只返回第一个众数，即使可能有多个

    # 将统计指标存储在字典中
    statistics = {
        # "ori": dl,
        "mean": mean_value,
        "variance": variance_value,
        "median": median_value,
        "mode": mode_value
    }
    print(statistics)
    return statistics


class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """

    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]  # [:100]
            stastic_ts(self.dori["smasks"])
        print(
            f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_" + key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]):
            pad_val (int, optional): pad value. Defaults to -1.
        Returns:
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)  # [0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        for i, row in df.iterrows():
            # use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(float(_)) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs"]:  # in ["smasks", "tseqs"]:
                dori[key] = torch.tensor(dori[key], dtype=torch.long)
            else:
                dori[key] = torch.tensor(dori[key], dtype=torch.float)

        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = torch.tensor(dqtest[key], dtype=torch.long)[:, 1:]

            return dori, dqtest
        return dori


# # used for spliting train_data and valid_data, and {0} is set as valid_data, {1-4} is set as train_data default.
# all_folds = {0, 1, 2, 3, 4}
#
# # get assist2009 dataset
# ast09_cur_valid = KTDataset("./data/assist2009/train_valid_sequences.csv", ["questions", "concepts"], {0})
# ast09_cur_train = KTDataset("./data/assist2009/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# ast09_valid_loader = DataLoader(ast09_cur_valid, batch_size=128)
# ast09_train_loader = DataLoader(ast09_cur_train, batch_size=128)

# # get assist2012 dataset
# ast12_cur_valid = KTDataset("./data/assist2012/train_valid_sequences.csv", ["questions", "concepts"], {0})
# ast12_cur_train = KTDataset("./data/assist2012/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# ast12_valid_loader = DataLoader(ast12_cur_valid, batch_size=128)
# ast12_train_loader = DataLoader(ast12_cur_train, batch_size=128)
#
# # # get assist2015 dataset
# # ast15_cur_valid = KTDataset("./data/assist2015/train_valid_sequences.csv", ["questions", "concepts"], {0})
# # ast15_cur_train = KTDataset("./data/assist2015/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# # ast15_valid_loader = DataLoader(ast15_cur_valid, batch_size=128)
# # ast15_train_loader = DataLoader(ast15_cur_train, batch_size=128)
#
# # get algebra2005 dataset
# al05_cur_valid = KTDataset("./data/algebra2005/train_valid_sequences.csv", ["questions", "concepts"], {0})
# al05_cur_train = KTDataset("./data/algebra2005/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# al05_valid_loader = DataLoader(al05_cur_valid, batch_size=128)
# al05_train_loader = DataLoader(al05_cur_train, batch_size=128)
#
# # get bridge2algebra2006 dataset
# bd06_cur_valid = KTDataset("./data/bridge2algebra2006/train_valid_sequences.csv", ["questions", "concepts"], {0})
# bd06_cur_train = KTDataset("./data/bridge2algebra2006/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# bd06_valid_loader = DataLoader(bd06_cur_valid, batch_size=128)
# bd06_train_loader = DataLoader(bd06_cur_train, batch_size=128)
#
# # get nips_task34 dataset
# nip34_cur_valid = KTDataset("./data/nips_task34/train_valid_sequences.csv", ["questions", "concepts"], {0})
# nip34_cur_train = KTDataset("./data/nips_task34/train_valid_sequences.csv", ["questions", "concepts"], all_folds - {0})
# nip34_valid_loader = DataLoader(nip34_cur_valid, batch_size=128)
# nip34_train_loader = DataLoader(nip34_cur_train, batch_size=128)
