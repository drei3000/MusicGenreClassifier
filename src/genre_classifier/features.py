import numpy as np
import librosa


def get_mfcc_features(segment: np.ndarray, n_fft: int = 2048, hop_length: int = 512, n_mfcc: int = 13) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=segment, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    feature_vector = []
    for i in range(mfccs.shape[0]):
        coeff = mfccs[i]
        feature_vector.extend([
            np.mean(coeff),
            np.std(coeff),
            np.max(coeff),
            np.min(coeff),
        ])
    return np.array(feature_vector)


