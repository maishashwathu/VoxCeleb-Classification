import os
import librosa
import numpy as np
from pydub import AudioSegment, effects
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random


def prepare_voxceleb_data(dataset_path):
    """
    Parses the VoxCeleb directory structure to create a file list.
    """
    print("\nScanning for audio files...")
    file_list = []
    for root, dirs, files in tqdm(os.walk(dataset_path), desc="Scanning directories"):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                path_parts = full_path.split(os.sep)
                speaker_id = path_parts[-3]
                file_list.append((full_path, speaker_id))
    print(f"Found {len(file_list)} audio files.")
    return file_list


def speed_perturbation(waveform, sr, speed_factor):
    """Apply speed perturbation to audio."""
    return librosa.effects.time_stretch(waveform, rate=speed_factor), sr


def add_noise(waveform, noise_level=0.005):
    """Add gaussian noise to waveform."""
    noise = np.random.randn(len(waveform)) * noise_level
    return waveform + noise


def pitch_shift(waveform, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


def apply_augmentations(waveform, sr, augment_prob=0.5):
    """Randomly apply data augmentations."""
    # Speed perturbation (most important for speaker recognition)
    if random.random() < augment_prob:
        speed_factor = random.choice([0.9, 1.1])  # 90% or 110% speed
        waveform, sr = speed_perturbation(waveform, sr, speed_factor)
    
    # Add background noise
    if random.random() < augment_prob:
        noise_level = random.uniform(0.002, 0.008)
        waveform = add_noise(waveform, noise_level)
    
    # Pitch shift (subtle)
    if random.random() < augment_prob:
        n_steps = random.choice([-1, 1])  # ±1 semitone
        waveform = pitch_shift(waveform, sr, n_steps)
    
    return waveform


def preprocess_audio(path, target_sr=16000, top_db=20, apply_augment=False):
    """Loads, cleans, and standardizes a single audio file."""
    try:
        waveform, sr = librosa.load(path, sr=None)
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
        
        # Apply data augmentation if enabled
        if apply_augment:
            waveform = apply_augmentations(waveform, target_sr)
        
        if np.max(np.abs(waveform)) > 0:
            audio_segment = AudioSegment(
                (waveform * 32767).astype(np.int16).tobytes(),
                frame_rate=target_sr,
                sample_width=2,
                channels=1
            )
            normalized_segment = effects.normalize(audio_segment)
            waveform = np.array(normalized_segment.get_array_of_samples()).astype(np.float32) / 32767.0
        return waveform, target_sr
    except Exception as e:
        print(f"ERROR processing {os.path.basename(path)}: {e}")
        return None, None

def spec_augment(mel_spec, freq_mask_param=15, time_mask_param=35, n_freq_masks=1, n_time_masks=1):
    """Apply SpecAugment: frequency and time masking on mel spectrogram."""
    mel_spec = mel_spec.copy()
    n_mels, n_frames = mel_spec.shape
    
    # Frequency masking
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, n_mels - f)
        mel_spec[f0:f0+f, :] = 0
    
    # Time masking
    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_param, n_frames - 1))
        t0 = random.randint(0, n_frames - t)
        mel_spec[:, t0:t0+t] = 0
    
    return mel_spec


class AudioPreprocessorDataset(Dataset):
    """Dataset to load, process audio, and convert to Mel spectrograms."""
    def __init__(self, file_list, apply_augment=False, apply_spec_augment=False):
        self.file_list = file_list
        self.apply_augment = apply_augment
        self.apply_spec_augment = apply_spec_augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio_path, _ = self.file_list[index]
        processed_waveform, sr = preprocess_audio(audio_path, apply_augment=self.apply_augment)
        if processed_waveform is None:
            return None, None
        mel_spectrogram = librosa.feature.melspectrogram(y=processed_waveform, sr=sr, n_mels=80)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Apply SpecAugment if enabled
        if self.apply_spec_augment:
            log_mel_spectrogram = spec_augment(log_mel_spectrogram)
        
        return torch.tensor(log_mel_spectrogram), audio_path

def collate_fn_filter_none(batch):
    """
    Custom collate function to filter out failed samples and pad spectrograms.
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    spectrograms, paths = zip(*batch)
    max_len = max(spec.shape[1] for spec in spectrograms)
    padded_spectrograms = [F.pad(spec, (0, max_len - spec.shape[1])) for spec in spectrograms]
    return torch.stack(padded_spectrograms, 0), list(paths)


if __name__ == "__main__":
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    # IMPORTANT: Update these paths to match your local machine's setup.
    
    # Path to the original VoxCeleb 'wav' directory.
    # Windows example: VOXCELEB_DEV_PATH = r"C:\data\VoxCeleb\wav"
    # macOS/Linux example: VOXCELEB_DEV_PATH = "/home/user/data/VoxCeleb/wav"
    VOXCELEB_DEV_PATH = r"D:\DL\Project\vox1_dev_wav\wav"  # <-- CHANGE THIS

    # Path where the processed .npy spectrogram files will be saved.
    # Windows example: PROCESSED_DATA_PATH = r"C:\data\processed_spectrograms"
    # macOS/Linux example: PROCESSED_DATA_PATH = "/home/user/data/processed_spectrograms"
    PROCESSED_DATA_PATH = r"D:\DL\Project\Spectograms" # <-- CHANGE THIS
    
    # --- AUGMENTATION SETTINGS ---
    APPLY_AUGMENTATION = False  # Enable waveform augmentation (speed, noise, pitch)
    APPLY_SPEC_AUGMENT = False  # Enable SpecAugment (frequency/time masking)

    # --- 2. RUN THE PIPELINE ---
    if "/path/to/your" in VOXCELEB_DEV_PATH or "/path/to/your" in PROCESSED_DATA_PATH:
        print("PLEASE UPDATE THE 'VOXCELEB_DEV_PATH' and 'PROCESSED_DATA_PATH' VARIABLES!")
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        # Step 1: Find all the .wav files
        all_files = prepare_voxceleb_data(VOXCELEB_DEV_PATH)
        
        # Step 2: Set up the Dataset and DataLoader
        preprocessor_dataset = AudioPreprocessorDataset(
            file_list=all_files,
            apply_augment=APPLY_AUGMENTATION,
            apply_spec_augment=APPLY_SPEC_AUGMENT
        )

        # For local machines, you can often use more workers for faster processing.
        # os.cpu_count() is a good starting point.
        num_workers = min(os.cpu_count(), 8) # Cap at 8 to avoid overwhelming system
        
        data_loader = DataLoader(
            dataset=preprocessor_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_filter_none
        )

        # Step 3: Loop, process, and save
        print(f"\nStarting pre-processing pipeline with {num_workers} workers...")
        for batch_spectrograms, batch_paths in tqdm(data_loader, desc="Processing and saving"):
            if batch_spectrograms is None:
                continue

            for i in range(len(batch_paths)):
                spectrogram = batch_spectrograms[i].numpy()
                original_path = batch_paths[i]

                # Recreate the original folder structure in the target directory
                relative_path = os.path.relpath(original_path, VOXCELEB_DEV_PATH)
                new_filename = os.path.splitext(relative_path)[0] + ".npy"
                target_path = os.path.join(PROCESSED_DATA_PATH, new_filename)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                np.save(target_path, spectrogram)

        print(f"\n Pre-processing complete! Spectrograms saved in '{PROCESSED_DATA_PATH}'")