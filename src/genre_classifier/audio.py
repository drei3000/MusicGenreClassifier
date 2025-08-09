import librosa
from typing import List


def segment_audio(file_path: str, segment_length_sec: float) -> List[list]:
    try:
        signal, sr = librosa.load(file_path, sr=22050)
        segment_samples = int(segment_length_sec * sr)
        segments = []
        for start in range(0, len(signal), segment_samples):
            end = start + segment_samples
            if end <= len(signal):
                segments.append(signal[start:end])
        return segments
    except Exception:
        return []


