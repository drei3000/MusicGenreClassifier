import os
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from .audio import segment_audio
from .features import get_mfcc_features


def create_dataset(
    data_dir: str,
    dataset_path: str,
    segment_length_sec: float = 10.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]]:
    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    features = []
    genres = []
    file_ids = []
    encoder = LabelEncoder()

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return None

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                file_path = os.path.join(genre_path, filename)
                segments = segment_audio(file_path, segment_length_sec)
                for segment in segments:
                    features.append(get_mfcc_features(segment))
                    genres.append(genre)
                    file_ids.append(file_path)

    if len(features) == 0:
        print("No audio segments found. Ensure data directory contains genre subfolders with audio files.")
        return None

    try:
        X = np.vstack(features)
    except ValueError:
        X = np.array(features)
        if X.ndim != 2:
            print("Inconsistent feature shapes; cannot assemble dataset.")
            return None

    Y = encoder.fit_transform(np.array(genres))
    groups = np.array(file_ids)

    print(f"Dataset created: {X.shape[0]} segments, {X.shape[1]} features")

    unique_group_count = len(np.unique(groups))
    if X.shape[0] < 2 or unique_group_count < 2:
        print("Not enough data to create a train/test split (need at least 2 files and 2 segments).")
        return None

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(gss.split(X, Y, groups=groups))

    features_train, features_test = X[train_index], X[test_index]
    genres_train, genres_test = Y[train_index], Y[test_index]
    file_ids_train, file_ids_test = groups[train_index], groups[test_index]

    print(
        f"Group-aware split -> train: {features_train.shape[0]} segments, "
        f"test: {features_test.shape[0]} segments, "
        f"unique train files: {len(np.unique(file_ids_train))}, "
        f"unique test files: {len(np.unique(file_ids_test))}"
    )

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    np.savez_compressed(
        dataset_path,
        features_train=features_train,
        features_test=features_test,
        genres_train=genres_train,
        genres_test=genres_test,
        file_ids_train=file_ids_train,
        file_ids_test=file_ids_test,
        label_classes=encoder.classes_,
    )
    print(f"Dataset saved to {dataset_path}")
    return features_train, features_test, genres_train, genres_test, encoder


def load_dataset(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    return (
        data['features_train'],
        data['features_test'],
        data['genres_train'],
        data['genres_test'],
        data['label_classes'],
    )


