import os
import pickle
import librosa
import numpy as np
from collections import Counter
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import librosa
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


def interactive_visualisation(model_path: str, dataset_path: str):
    """
    Create a simple, robust interactive visualization showing genre spectrograms.
    """
    # Load data and model
    data = np.load(dataset_path, allow_pickle=True)
    features_train = data['features_train']
    features_test = data['features_test']
    genres_train = data['genres_train']
    genres_test = data['genres_test']
    label_classes = data['label_classes']
    file_ids_train = data['file_ids_train']
    file_ids_test = data['file_ids_test']
    
    model, encoder = load_trained_model(model_path)
    if model is None:
        print("Error: Could not load model")
        return
    
    # Pre-compute spectrograms for all genres to avoid real-time processing
    print("Pre-computing spectrograms for all genres...")
    genre_spectrograms = {}
    
    for genre_idx, genre_name in enumerate(label_classes):
        genre_mask = genres_test == genre_idx
        genre_file_ids = file_ids_test[genre_mask]
        unique_files = np.unique(genre_file_ids)
        
        if len(unique_files) == 0:
            continue
        
        spectrograms = []
        for file_path in unique_files[:2]:  # Use fewer files for speed
            try:
                y, sr = librosa.load(file_path, sr=22050, duration=5.0)  # Shorter duration
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)  # Fewer mels
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                spectrograms.append(mel_spec_db)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if spectrograms:
            avg_spec = np.mean(spectrograms, axis=0)
            genre_spectrograms[genre_name] = avg_spec
            print(f"* {genre_name}: {len(spectrograms)} files")
    
    if not genre_spectrograms:
        print("No spectrograms could be computed. Check your audio files.")
        return
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    # Create a beautiful grid of spectrograms
    n_genres = len(genre_spectrograms)
    cols = min(3, n_genres)
    rows = (n_genres + cols - 1) // cols
    
    # Create figure with better proportions
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows), 
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    
    # Handle single row/column cases
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define a beautiful colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot spectrograms with enhanced styling
    for idx, (genre_name, spec) in enumerate(genre_spectrograms.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Use a more appealing colormap
        im = ax.imshow(spec, aspect='auto', origin='lower', 
                      cmap='magma', interpolation='bilinear')
        
        # Enhanced title with genre name
        ax.set_title(f'{genre_name.upper()}', fontsize=14, fontweight='bold', 
                    color=colors[idx % len(colors)], pad=15)
        
        # Better axis labels
        ax.set_xlabel('Time (frames)', fontsize=11, fontweight='semibold')
        ax.set_ylabel('Mel Frequency', fontsize=11, fontweight='semibold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Power (dB)', fontsize=10, fontweight='semibold')
        cbar.ax.tick_params(labelsize=9)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Better tick styling
        ax.tick_params(axis='both', which='major', labelsize=9, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Hide empty subplots
    for idx in range(len(genre_spectrograms), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    # Enhanced main title
    fig.suptitle('MUSIC GENRE SPECTROGRAM ANALYSIS', fontsize=20, fontweight='bold', 
                y=0.98, color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.95, 'Average Mel Spectrograms by Genre - Audio Frequency Patterns', 
             fontsize=14, ha='center', style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    plt.show()
    
    # Also show feature correlation heatmap with enhanced styling
    print("\nCreating feature correlation heatmap...")
    all_features = np.vstack([features_train, features_test])
    corr_matrix = np.corrcoef(all_features.T)
    
    # Create a beautiful correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a better colormap for correlation
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1,
                   interpolation='nearest')
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='semibold')
    cbar.ax.tick_params(labelsize=10)
    
    # Better title and labels
    ax.set_title('MFCC Feature Correlation Matrix', fontsize=16, fontweight='bold', 
                pad=20, color='#2c3e50')
    ax.set_xlabel('Feature Index', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Feature Index', fontsize=12, fontweight='semibold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Better tick styling
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add text annotation for interpretation
    fig.text(0.5, 0.02, 'Red: Positive correlation | Blue: Negative correlation | White: No correlation', 
             fontsize=11, ha='center', style='italic', color='#7f8c8d',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")

