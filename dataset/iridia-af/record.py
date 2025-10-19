from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def create_record(record_id, metadata_df, record_path):
    metadata_record = (metadata_df[metadata_df["record_id"] == record_id])
    assert len(metadata_record) == 1
    metadata_record = metadata_record.values[0]
    record_path = Path(record_path, record_id)
    record = Record(record_path, metadata_record)
    return record


class Record:
    def __init__(self, record_folder, metadata_record):
        # self.metadata = self.__get_metadata(record_id)
        self.metadata = RecordMetadata(*metadata_record)

        self.record_folder = record_folder
        self.num_days = len(list(self.record_folder.glob("*ecg_*.h5")))

        self.rr_files = sorted(self.record_folder.glob("*rr_*.h5"))
        assert len(self.rr_files) == self.num_days
        self.rr = None
        self.rr_labels_df = None
        self.rr_labels = None

        self.ecg_files = sorted(self.record_folder.glob("*ecg_*.h5"))
        assert len(self.ecg_files) == self.num_days
        self.ecg = None
        self.ecg_labels_df = None
        self.ecg_labels = None

    def load_ecg(self, clean_front=False):
        ecg_files = sorted(self.record_folder.glob("*_ecg_*.h5"))
        self.ecg = [self.__read_ecg_file(ecg_file, clean_front) for ecg_file in ecg_files]
        self.__create_ecg_labels(clean_front)

    def __read_ecg_file(self, ecg_file: Path, clean_front=False) -> np.ndarray:
        with h5py.File(ecg_file, "r") as f:
            key = list(f.keys())[0]
            ecg = f[key][:]
            if clean_front:
                ecg = ecg[6000:]
        return ecg

    def __read_ecg_labels(self) -> pd.DataFrame:
        ecg_labels = sorted(self.record_folder.glob("*ecg_labels.csv"))
        df_ecg_labels = pd.read_csv(ecg_labels[0])
        return df_ecg_labels

    def __create_ecg_labels(self, clean_front=False):
        self.ecg_labels_df = self.__read_ecg_labels()
        len_ecg = [len(ecg) for ecg in self.ecg]

        start_day = self.ecg_labels_df["start_file_index"].unique()
        end_day = self.ecg_labels_df["end_file_index"].unique()
        days = np.unique(np.concatenate([start_day, end_day]))
        assert len(days) <= len(len_ecg)

        labels = [np.zeros(len_day_ecg) for len_day_ecg in len_ecg]

        for i, row in self.ecg_labels_df.iterrows():
            if row.start_file_index == row.end_file_index:
                labels[row.start_file_index][row.start_qrs_index:row.end_qrs_index] = 1
            else:
                labels[row.start_file_index][row.start_qrs_index:] = 1
                labels[row.end_file_index][:row.end_qrs_index] = 1
                if row.end_file_index - row.end_file_index > 1:
                    for day in range(row.start_file_index + 1, row.end_file_index):
                        labels[day][:] = 1
        if clean_front:
            labels = [label[6000:] for label in labels]
        self.ecg_labels = labels

    def plot_ecg(self, has_day_ticks=True):
        all_ecg = np.concatenate(self.ecg)
        all_ecg_labels = np.concatenate(self.ecg_labels)

        # set font size
        plt.rcParams.update({"font.size": 18})

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax[0].plot(all_ecg[:, 0])
        ax[0].set_ylabel("ECG I (mV)")

        ax[1].plot(all_ecg[:, 1])
        ax[1].set_ylabel("ECG II (mV)")

        ax[2].plot(all_ecg_labels)
        ax[2].set_ylim(-0.1, 1.1)
        ax[2].set_yticks([0, 1])
        ax[2].set_yticklabels(["NSR", "AF"])
        ax[2].set_xlabel("ECG index", labelpad=10)
        ax[2].set_ylabel("Label")

        if has_day_ticks:
            # add vertical lines at the end of each day
            len_ecg = [len(ecg) for ecg in self.ecg]
            index_len_ecg = np.cumsum(len_ecg)
            index_len_ecg = np.insert(index_len_ecg, 0, 0)
            for i in index_len_ecg:
                ax[0].axvline(i, color="k", linestyle="--", alpha=0.5)
                ax[1].axvline(i, color="k", linestyle="--", alpha=0.5)
                ax[2].axvline(i, color="k", linestyle="--", alpha=0.5)

        plt.show()

    def number_of_episodes(self):
        rr_labels = sorted(self.record_folder.glob("*rr_labels.csv"))
        df_rr_labels = pd.read_csv(rr_labels[0])
        num_episodes_rr = len(df_rr_labels)

        ecg_labels = sorted(self.record_folder.glob("*ecg_labels.csv"))
        df_ecg_labels = pd.read_csv(ecg_labels[0])
        num_episodes_ecg = len(df_ecg_labels)

        assert num_episodes_rr == num_episodes_ecg
        return num_episodes_rr


@dataclass
class RecordMetadata:
    patient_id: str
    patient_sex: str
    patient_age: int
    record_id: str
    record_date: str
    record_start_time: str
    record_end_time: str
    record_timedelta: str
    record_n_files: int
    record_n_seconds: int
    record_n_samples: int


@dataclass
class ECGEvent:
    # start event
    start_datetime: str
    start_file_index: int
    start_qrs_index: int
    # end event
    end_datetime: str
    end_file_index: int
    end_qrs_index: int
    # duration
    af_duration: int
    nsr_duration: int


@dataclass
class RREvent:
    # start event
    start_file_index: int
    start_rr_index: int
    # end event
    end_file_index: int
    end_rr_index: int
