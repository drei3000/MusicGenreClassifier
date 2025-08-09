import os
import pickle
import numpy as np
from collections import Counter
from typing import Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .dataset import load_dataset


def train_model(dataset_path: str, model_save_path: str):
    features_train, features_test, genres_train, genres_test, label_classes = load_dataset(dataset_path)

    print(f"Training on {features_train.shape[0]} samples...")
    print(f"Features per sample: {features_train.shape[1]}")
    print(f"Number of genres: {len(label_classes)}")

    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(features_train, genres_train)

    train_accuracy = rf_model.score(features_train, genres_train)
    test_accuracy = rf_model.score(features_test, genres_test)

    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_data = {
        'model': rf_model,
        'encoder_classes': label_classes,
        'dataset_path': dataset_path,
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model and encoder saved to {model_save_path}")
    return rf_model


def load_trained_model(model_path: str) -> Tuple[Optional[object], Optional[LabelEncoder]]:
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict):
            model = model_data['model']
            encoder_classes = model_data['encoder_classes']
            encoder = LabelEncoder()
            encoder.classes_ = encoder_classes
        else:
            model = model_data
            encoder = None
        print(f"Model loaded from {model_path}")
        return model, encoder
    except FileNotFoundError:
        print(f"No saved model found at {model_path}")
        return None, None


def evaluate_on_test_set(model_path: str, dataset_path: str, level: str = "file"):
    data = np.load(dataset_path, allow_pickle=True)
    X_test = data['features_test']
    y_test = data['genres_test']
    label_classes = data['label_classes']
    file_ids_test = data.get('file_ids_test', None)

    model, _ = load_trained_model(model_path)
    if model is None:
        print("Model not found")
        return

    y_pred_seg = model.predict(X_test)

    if level == "segment" or file_ids_test is None:
        acc = accuracy_score(y_test, y_pred_seg)
        print(f"Segment-level accuracy: {acc:.3f}")
        cm = confusion_matrix(y_test, y_pred_seg, labels=np.arange(len(label_classes)))
        disp = ConfusionMatrixDisplay(cm, display_labels=label_classes)
        disp.plot(xticks_rotation=45, cmap='Blues')
        plt.title('Confusion Matrix (segments)')
        plt.tight_layout()
        plt.show()
        return

    unique_files = np.unique(file_ids_test)
    y_true_files = []
    y_pred_files = []
    for f in unique_files:
        idx = np.where(file_ids_test == f)[0]
        true_label = int(np.bincount(y_test[idx]).argmax())
        votes = y_pred_seg[idx]
        pred_label = int(np.bincount(votes).argmax())
        y_true_files.append(true_label)
        y_pred_files.append(pred_label)

    y_true_files = np.array(y_true_files)
    y_pred_files = np.array(y_pred_files)
    acc_files = accuracy_score(y_true_files, y_pred_files)
    print(f"File-level accuracy: {acc_files:.3f}  (num files: {len(unique_files)})")

    cm_files = confusion_matrix(y_true_files, y_pred_files, labels=np.arange(len(label_classes)))
    disp = ConfusionMatrixDisplay(cm_files, display_labels=label_classes)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title('Confusion Matrix (files)')
    plt.tight_layout()
    plt.show()


def get_encoder_from_dataset(dataset_path: str) -> Optional[LabelEncoder]:
    try:
        data = np.load(dataset_path)
        label_classes = data['label_classes']
        encoder = LabelEncoder()
        encoder.classes_ = label_classes
        return encoder
    except Exception as e:
        print(f"Error loading encoder from dataset: {e}")
        return None


def predict_single_file(audio_file_path: str, model_path: str, dataset_path: str):
    try:
        print(f"Analyzing:  {audio_file_path}")
        if not os.path.exists(audio_file_path):
            print(f"Error: File {audio_file_path} not found")
            return None, None

        # Default to 10s segments for prediction
        import librosa
        from .audio import segment_audio
        from .features import get_mfcc_features

        segments = segment_audio(audio_file_path, 10)
        print(f"Found {len(segments)} segments")
        if len(segments) == 0:
            print("Error: No audio segments could be extracted from the file")
            return None, None

        segment_features = [get_mfcc_features(s) for s in segments]
        features_array = np.vstack(segment_features)
        print(f"Features array shape: {features_array.shape}")

        model, _ = load_trained_model(model_path)
        if model is None:
            print("Error: Could not load model")
            return None, None

        encoder = get_encoder_from_dataset(dataset_path)
        if encoder is None:
            print("Error: Could not load encoder from dataset")
            return None, None

        print("Model and encoder loaded successfully")
        print("Making predictions...")
        predictions = model.predict(features_array)

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)
            avg_probabilities = np.mean(probabilities, axis=0)
        else:
            avg_probabilities = None

        vote_counts = Counter(predictions)
        predicted_genre_encoded = vote_counts.most_common(1)[0][0]
        predicted_genre = encoder.classes_[predicted_genre_encoded]
        confidence = vote_counts[predicted_genre_encoded] / len(predictions)

        print("\nResults:")
        print(f"Predicted Genre: {predicted_genre}")
        print(f"Confidence: {confidence:.3f} ({vote_counts[predicted_genre_encoded]}/{len(predictions)} segments)")

        if avg_probabilities is not None:
            print("\nGenre Probabilities:")
            for i, genre in enumerate(encoder.classes_):
                print(f"  {genre}: {avg_probabilities[i]:.3f}")

        return predicted_genre, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


