import warnings

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn import metrics


def preprocess_recording(recording, fs, standardize=False):
    recording_m = recording - recording.mean(axis=1, keepdims=True)
    recording = butter_highpass_filter(recording_m, 0.5, fs)
    recording = recording - recording.mean(axis=1, keepdims=True)
    if standardize:
        recording = recording / (recording.std(axis=1, keepdims=True)+ 1e-6)
    return recording

# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_highpass(lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, lowcut, fs, order=1):
    sos = butter_highpass(lowcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def is_noisy_window(recording: np.ndarray, fs: int) -> bool:
    recording_m = recording - np.mean(recording)
    recording_f = butter_highpass_filter(recording_m, 0.5, fs)
    abs_recording = np.absolute(recording_f)
    if np.any(abs_recording > 5) or np.all(abs_recording < 0.03):
        return True
    else:
        return False

def is_low_signal_window(recording: np.ndarray, fs: int) -> bool:
    recording_m = recording - np.mean(recording)
    recording_f = butter_highpass_filter(recording_m, 0.5, fs)
    abs_recording = np.absolute(recording_f)
    if np.all(abs_recording < 0.03):
        return True
    else:
        return False

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float=None):
    num_valid = len(labels)
    if threshold is None:
        threshold = compute_threshold(labels, probs)
    preds = probs > threshold

    # Accuracy
    num_corrrects = (preds == labels).sum()
    accuracy = num_corrrects / num_valid

    # Sensitivity, Specificity, Precision, NPV, F1-score
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    # AUROC, AUPRC
    roc_auc_score = metrics.roc_auc_score(labels, probs)
    precisions, recalls, thresholds = metrics.precision_recall_curve(labels, probs)
    auc_prc_score = metrics.auc(recalls, precisions)
    auprc_baseline = labels.sum() / num_valid

    metrics_dict = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1_score": f1_score,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "auprc_baseline": auprc_baseline,
        "auroc": roc_auc_score,
        "auprc": auc_prc_score,
        "threshold": threshold,
    }

    return metrics_dict

def compute_threshold(labels, probs):
    # Find a threshold that maximizes Youden's J statistics
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    J = tpr-fpr
    threshold_idx = np.argmax(J[1:])+1 # not to choose thresholds[0], which is np.inf
    threshold = thresholds[threshold_idx]
    return threshold
