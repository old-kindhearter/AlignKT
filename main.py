import os
import json
import time
import torch
import random
import argparse
import numpy as np
from train import train_model
from data_loader import KTDataset
from torch.utils.data import DataLoader

from model.AlignKT import AlignKT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model info
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="./saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)  # used for spliting train_data and valid_data

    # model hyper-parameters
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--drop_out", type=float, default=0.25)
    parser.add_argument("--n_blocks", type=int, default=1)

    # training info
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)

    args = parser.parse_args()
    params = vars(args)

    # set seed
    random.seed(params["seed"])
    os.environ['PYTHONHASHSEED'] = str(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.cuda.manual_seed(params["seed"])
    torch.cuda.manual_seed_all(params["seed"])
    torch.backends.cudnn.deterministic = True

    # initialize dataset, read specific dataset info from data_config.json
    def read_json(file_path):
        with open(file_path, 'r') as jf:
            data = json.load(jf)
        return data

    dataset_config = read_json("./config/data_config.json")
    dataset_name = params['dataset_name']
    specific_dataset_conf = dataset_config[dataset_name]

    # load train_data and valid_data
    # uesd for spliting train_data and valid_data, and {o} is set as valid_data, f1-4} is set as train_data default.
    all_folds = {0, 1, 2, 3, 4}
    data_set_path = specific_dataset_conf['dpath'][1:]+'/'+specific_dataset_conf['train_valid_file']  # modify to adjust to local dataset path
    valid_data_loader = DataLoader(KTDataset(data_set_path, specific_dataset_conf['input_type'], {0}), batch_size=params['batch_size'])
    train_data_loader = DataLoader(KTDataset(data_set_path, specific_dataset_conf['input_type'], all_folds-{0}), batch_size=params['batch_size'])

    # load test_data and test_window_data(concept level). used for evaluation
    test_set_path = specific_dataset_conf['dpath'][1:]+'/'+specific_dataset_conf['test_file']
    test_data_loader = DataLoader(KTDataset(test_set_path, specific_dataset_conf['input_type'], {-1}),
                                  batch_size=params['batch_size'], shuffle=False)
    # test_window_set_path = specific_dataset_conf['dpath'][1:]+'/'+specific_dataset_conf['test_window_file']
    # test_window_data_loader = DataLoader(KTDataset(test_window_set_path, specific_dataset_conf['input_type'], {-1}),
    #                                      batch_size=params['batch_size'], shuffle=False)

    # load test_data and test_window_data(question level). used for evaluation
    testq_set_path = specific_dataset_conf['dpath'][1:]+'/'+specific_dataset_conf['test_question_file']
    testq_data_loader = DataLoader(KTDataset(testq_set_path, specific_dataset_conf['input_type'], {-1}, True),
                                  batch_size=params['batch_size'], shuffle=False)
    # testq_window_set_path = specific_dataset_conf['dpath'][1:]+'/'+specific_dataset_conf['test_question_window_file']
    # testq_window_data_loader = DataLoader(KTDataset(testq_window_set_path, specific_dataset_conf['input_type'], {-1}, True),
    #                                      batch_size=params['batch_size'], shuffle=False)

    model = AlignKT(params['d_model'], params['n_heads'], params['d_ff'], params['drop_out'], params['n_blocks'], **specific_dataset_conf)
  
    print(model)
    print("start Training...")

    start_time = time.time()
    train_model(model, train_data_loader, valid_data_loader, test_data_loader, testq_data_loader, **params)
    end_time = time.time()
    print("training time: {:.2f} sec.".format(end_time - start_time))
