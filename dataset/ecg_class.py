import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


class ECGEpisode: # default: 5 min
    def __init__(self, patient_dir, recording, time_info, label, sampling_rate, beat_df: pd.DataFrame):
        self.patient_dir = patient_dir
        self.recording = recording
        self.span_idx = time_info[0]
        self.start = time_info[1]
        self.end = time_info[2]
        self.label = label
        self.sampling_rate = sampling_rate
        self.beat_df = beat_df
        
        self.windows = {}
        self.window_sec = None
        self.stride_sec = None
        
    def build_windows(self, window_sec: int, stride_sec: int):
        # window: seconds, stride: seconds
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        window_points = self.window_sec * self.sampling_rate
        stride_points = self.stride_sec * self.sampling_rate
        
        for block_number, window_end in enumerate(range(self.end - self.start, window_points-1, -stride_points)):
            window_str = f"b{block_number}"
            window_start = window_end - window_points
            window_recording = self.recording[window_start:window_end]
            window_beat_df = self.beat_df[(self.beat_df.BeatRow >= self.start+window_start) & (self.beat_df.BeatRow < self.start+window_end)]
            window = ECGWindow(self.patient_dir, window_recording, (self.span_idx, self.start+window_start, self.start+window_end), self.label, self.sampling_rate, window_beat_df, window_str, f"{self.patient_dir}#{self.start}")
            window.calculate_features()
            self.windows[window_str] = window
            
            
class ECGWindow: # default: 60 secs
    def __init__(self, patient_dir, recording, time_info, label, sampling_rate, beat_df: pd.DataFrame, window_str, episode_str):
        self.patient_dir = patient_dir
        self.recording = recording
        self.span_idx = time_info[0]
        self.start = time_info[1]
        self.end = time_info[2]
        self.label = label
        self.sampling_rate = sampling_rate
        
        self.window_str = window_str
        self.episode_str = episode_str
        
        self.beat_df = beat_df
        self.features = {"heart_rate": None, "rmssd": None, "rr_intervals": None, "pac_burden": None, "pvc_burden": None}

    def calculate_features(self):
        preprocessed_recording = preprocess_recording(self.recording[np.newaxis, ...], self.sampling_rate).squeeze()
        self.features["heart_rate"], self.features["rmssd"], self.features["rr_intervals"] = calculate_hrv_features(preprocessed_recording, self.sampling_rate)
        self.features["pac_burden"], self.features["pvc_burden"] = calculate_beat_burden(self.beat_df)

# /src/utils.py
def preprocess_recording(recording, fs=250): # process one by one
    recording_m = recording - np.mean(recording)
    recording_f = butter_highpass_filter(recording_m, 0.5, fs)
    return recording_f

def butter_highpass(lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, lowcut, fs, order=1):
    sos = butter_highpass(lowcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def calculate_hrv_features(recording, sampling_rate):
    try: 
        peaks, info = nk.ecg_peaks(recording, sampling_rate=sampling_rate)
    except:
        return float('nan'), float('nan'), np.array([], dtype=np.int64)
    
    try:
        hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate)
    except:
        heart_rate, rmssd = float('nan'), float('nan')
    else:
        rmssd = hrv_time.HRV_RMSSD[0]
        mean_rr = hrv_time.HRV_MeanNN[0] # unit: miliseconds
        heart_rate = 60000 / mean_rr # 1 minute / mean_rr
    
    r_peaks_array = info["ECG_R_Peaks"]
    if len(r_peaks_array) > 1:
        rr_intervals = r_peaks_array[1:] - r_peaks_array[:-1]
    else:
        rr_intervals = np.array([], dtype=np.int64)
        
    return heart_rate, rmssd, rr_intervals
    
def calculate_beat_burden(beat_df: pd.DataFrame):
    if len(beat_df) == 0:
        return float('nan'), float('nan')

    beat_counts = beat_df.BeatClass.value_counts()

    pac_burden = beat_counts['S'] / sum(beat_counts.values) if 'S' in beat_counts.keys() else 0
    pvc_burden = beat_counts['V'] / sum(beat_counts.values) if 'V' in beat_counts.keys() else 0

    return pac_burden, pvc_burden
