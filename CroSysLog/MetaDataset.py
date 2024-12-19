import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pandas as pd
import numpy as np
import re
from transformers import BertModel, BertTokenizer, BertConfig
import os
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from NeuralParser import BertEmbeddings


class MetaDataset(Dataset):
    def __init__(self, n_task, target_n_task, k_spt, k_query, source_system, target_system, window_size):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.current_directory = os.getcwd()
        # '/projappl/project_2006059/mixAEMeta'
        print("Current directory's absolute path:", self.current_directory)
        self.dataset_directory = '/projappl/project_2006059/datasets'

        self.n_task = int(n_task)  # batch of set args.task_num #
        self.target_n_task = target_n_task
        # self.n_way = n_way  # n-way
        self.k_spt = k_spt  # k-shot
        self.k_qry = k_query  # for evaluation

        # self.setsz = self.n_way * self.k_spt  # num of samples per set
        # self.querysz = self.n_way * self.k_qry  # number of samples per set for evaluation

        self.window_size = window_size
        self.source_system = source_system
        self.target_system = target_system
        self.feature_dim = 768

        self.NeuralParser = BertEmbeddings()

        self.source_system = source_system
        # self.data_frames = self.load_datasets()
        self.target_sys = pd.read_csv(f"{self.dataset_directory}/{self.target_system}/train_{self.target_system}.csv")

        #print("dfdfd", flush=True)

    def compare_and_switch(self, spt_starts, qry_starts):
        for i in range(min(len(spt_starts), len(qry_starts))):
            if qry_starts[i] < spt_starts[i]:
                # Switch the elements
                spt_starts[i], qry_starts[i] = qry_starts[i], spt_starts[i]
        return spt_starts, qry_starts

    def select_start_indexes_with_anomalies(self, n_tasks, df, k_spt, k_qry, spt_anomaly_ratio_lower=0.1,
                                            spt_anomaly_ratio_upper=0.5,
                                            qry_anomaly_ratio_lower=0.01, qry_anomaly_ratio_upper=0.5):

        np.random.seed(42)
        df_len = len(df)
        max_spt_start_index = df_len - k_spt
        max_qry_start_index = df_len - k_qry

        valid_spt_starts = []
        valid_qry_starts = []

        anomaly_counts_spt = df['anomaly'].rolling(window=k_spt, min_periods=1).sum()
        anomaly_counts_qry = df['anomaly'].rolling(window=k_qry, min_periods=1).sum()

        for start_index in range(max_spt_start_index + 1):
            anomaly_count = anomaly_counts_spt.iloc[start_index + k_spt - 1]
            if anomaly_count >= spt_anomaly_ratio_lower * k_spt and anomaly_count <= spt_anomaly_ratio_upper * k_spt:
                valid_spt_starts.append(start_index)

        for start_index in range(max_qry_start_index + 1):
            anomaly_count = anomaly_counts_qry.iloc[start_index + k_qry - 1]
            if anomaly_count >= qry_anomaly_ratio_lower * k_qry and anomaly_count <= qry_anomaly_ratio_upper * k_qry:
                valid_qry_starts.append(start_index)

        if not valid_spt_starts or not valid_qry_starts:
            raise ValueError("No valid start index combination found that meets the anomaly percentage criteria.")

        spt_starts = []
        qry_starts = []
        for _ in range(n_tasks):
            tem_spt_start = np.random.choice(valid_spt_starts)
            spt_starts.append(tem_spt_start)
            valid_spt_starts.remove(tem_spt_start)

            valid_spt_starts = [start for start in valid_spt_starts if
                               all(start >= spt + k_spt or start + k_spt <= spt for spt in spt_starts)]

            # select one qry_start
            tem_qry_start = np.random.choice(valid_qry_starts)
            qry_starts.append(tem_qry_start)
            valid_qry_starts.remove(tem_qry_start)

            valid_qry_starts = [start for start in valid_qry_starts if
                                all(start >= qry + k_qry or start + k_qry <= qry for qry in qry_starts)]

        # Call the function and print the modified lists
        # modified_spt_starts, modified_qry_starts = self.compare_and_switch(spt_starts, qry_starts)
        print("spt_starts:", spt_starts)
        print("qry_starts:", qry_starts)
        return spt_starts, qry_starts

    def load_data_cache_source(self, source_system):

        self.n_window_spt = int(self.k_spt / self.window_size)
        self.n_window_qry = int(self.k_qry / self.window_size)

        source_sys_df = pd.read_csv(f"{self.dataset_directory}/{source_system}/train_{source_system}.csv")
        x_spt_tensor = torch.zeros(self.n_task, self.k_spt, self.feature_dim)
        y_spt_tensor = torch.zeros(self.n_task, self.k_spt)
        x_qry_tensor = torch.zeros(self.n_task, self.k_qry, self.feature_dim)
        y_qry_tensor = torch.zeros(self.n_task, self.k_qry)

        spt_start, qry_start = self.select_start_indexes_with_anomalies(n_tasks=self.n_task, df=source_sys_df,
                                                                        k_spt=self.k_spt,
                                                                        k_qry=self.k_qry,
                                                                        spt_anomaly_ratio_lower=0.01,
                                                                        spt_anomaly_ratio_upper=0.5,
                                                                        qry_anomaly_ratio_lower=0.01,
                                                                        qry_anomaly_ratio_upper=0.5)
        for task_num in range(self.n_task):
            spt_start_idx = spt_start[task_num]
            qry_start_idx = qry_start[task_num]
            spt_set_df = source_sys_df.iloc[spt_start_idx:spt_start_idx + self.k_spt].reset_index(drop=True)
            qry_set_df = source_sys_df.iloc[qry_start_idx:qry_start_idx + self.k_qry].reset_index(drop=True)
            print("check", flush=True)

            x_spt_tensor[task_num, :, :], y_spt_tensor[task_num, :] = self.create_log_vec(spt_set_df,
                                                                                          sys=source_system)
            x_qry_tensor[task_num, :, :], y_qry_tensor[task_num, :] = self.create_log_vec(qry_set_df,
                                                                                          sys=source_system)
        reshape_x_spt_tensor = x_spt_tensor.view(self.n_task, self.n_window_spt, self.window_size, self.feature_dim)
        reshape_y_spt_tensor = y_spt_tensor.view(self.n_task, self.n_window_spt, self.window_size)
        reshape_x_qry_tensor = x_qry_tensor.view(self.n_task, self.n_window_qry, self.window_size, self.feature_dim)
        reshape_y_qry_tensor = y_qry_tensor.view(self.n_task, self.n_window_qry, self.window_size)
        return reshape_x_spt_tensor, reshape_y_spt_tensor, reshape_x_qry_tensor, reshape_y_qry_tensor

    def load_data_cache_target(self):
        print("source_system:", self.target_system)
        self.k_spt = 1000
        self.n_window_spt = int(self.k_spt / self.window_size)
        self.n_window_qry = int(self.k_qry / self.window_size)

        x_spt_tensor = torch.zeros(self.target_n_task, self.k_spt, self.feature_dim)
        y_spt_tensor = torch.zeros(self.target_n_task, self.k_spt)
        x_qry_tensor = torch.zeros(self.target_n_task, self.k_qry, self.feature_dim)
        y_qry_tensor = torch.zeros(self.target_n_task, self.k_qry)

        spt_start, qry_start = self.select_start_indexes_with_anomalies(n_tasks=self.target_n_task, df=self.target_sys,
                                                                        k_spt=self.k_spt, k_qry=self.k_qry,
                                                                        spt_anomaly_ratio_lower=0.2,
                                                                        spt_anomaly_ratio_upper=0.5,
                                                                        qry_anomaly_ratio_lower=0.01,
                                                                        qry_anomaly_ratio_upper=0.5)
        for task_num in range(self.target_n_task):
            spt_start_idx = spt_start[task_num]
            qry_start_idx = qry_start[task_num]
            spt_set_df = self.target_sys.iloc[spt_start_idx:spt_start_idx + self.k_spt].reset_index(drop=True)
            qry_set_df = self.target_sys.iloc[qry_start_idx:qry_start_idx + self.k_qry].reset_index(drop=True)
            print("check", flush=True)

            x_spt_tensor[task_num, :, :], y_spt_tensor[task_num, :] = self.create_log_vec(spt_set_df,
                                                                                          sys=self.target_system)
            x_qry_tensor[task_num, :, :], y_qry_tensor[task_num, :] = self.create_log_vec(qry_set_df,
                                                                                          sys=self.target_system)

        reshape_x_spt_tensor = x_spt_tensor.view(self.target_n_task, self.n_window_spt, self.window_size,
                                                 self.feature_dim)
        reshape_y_spt_tensor = y_spt_tensor.view(self.target_n_task, self.n_window_spt, self.window_size)
        reshape_x_qry_tensor = x_qry_tensor.view(self.target_n_task, self.n_window_qry, self.window_size,
                                                 self.feature_dim)
        reshape_y_qry_tensor = y_qry_tensor.view(self.target_n_task, self.n_window_qry, self.window_size)

        return reshape_x_spt_tensor, reshape_y_spt_tensor, reshape_x_qry_tensor, reshape_y_qry_tensor

    def create_log_vec(self, df, sys):
        min_max_scaler = MinMaxScaler()
        ano_label = df['anomaly'].astype(int).tolist()
        ano_label_tensor = torch.tensor(ano_label)

        df['normalized_timestamp'] = min_max_scaler.fit_transform(df[['timestamp']])
        selected_features = ['normalized_timestamp']
        feature_values_list = df[selected_features].values.tolist()
        combined_feature_tensor = torch.tensor(feature_values_list)
        # Move the tensor to the device
        # combined_feature_tensor = combined_feature_tensor.to(self.device)

        logMsgs = df['LogMsgFull'].tolist()

        logMsgs_emb = self.NeuralParser.create_bert_emb(logMsgs, sys)
        log_tensor = torch.cat((combined_feature_tensor, logMsgs_emb), dim=1)
        return logMsgs_emb, ano_label_tensor
