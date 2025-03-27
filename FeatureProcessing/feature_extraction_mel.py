"""
Extract Mel spectrogram features from audio files and save them to a CSV file.
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
from tqdm.auto import tqdm

print("Audio Analysis with Python")


audio_files = glob.glob(
    "./generated/characters_wav_files_44100Hz/generated_combined/**/*.wav", recursive=True
)
print(f"Number of audio files: {len(audio_files)}")

voice_profiles = pd.read_csv("./outputs/voices.csv")  # Load the voice profiles
# get only en- locale voices
voice_profiles = voice_profiles[voice_profiles["locale"].str.contains("en-")]
# drop voice_type and style_list columns
voice_profiles.drop(columns=["voice_type", "style_list"], inplace=True)
# print(voice_profiles.head())
# ["locale", "local_name", "short_name", "gender"]


# Function to calculate the mean and std for each Mel band over an entire audio file
def extract_mel_features(file_path, sample_rate=16000, n_mels=128):
    waveform, sr = torchaudio.load(file_path)

    # Resample to the target sample rate if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Extract Mel spectrogram
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="htk",
        win_length=None,
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to decibels (log scale)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    mel_features = mel_spectrogram_db

    # Compute mean and std for each Mel band (across time frames)
    mel_mean = torch.mean(mel_features, dim=-1).squeeze().numpy()  # Mean along time axis
    mel_std = torch.std(mel_features, dim=-1).squeeze().numpy()  # Std along time axis

    return mel_mean, mel_std, mel_features.squeeze().numpy()


n_mels = 128
feature_stats = []
feature_names = []

for audio_file in tqdm(audio_files):
    pitch_shifted = 100
    speed_changed = 100
    short_name = Path(audio_file).parent.stem

    if "pitch_shifted" in audio_file:
        short_name, pitch_shifted = Path(audio_file).stem.split("_pitch_shifted_")

    if "speed_changed" in audio_file:
        short_name, speed_changed = Path(audio_file).stem.split("_speed_changed_")

    row = voice_profiles.loc[voice_profiles["short_name"] == short_name].values[0]

    mel_mean, mel_std, _ = extract_mel_features(audio_file)

    # Prepare a single row with mean and std for current file
    features_row = np.concatenate((mel_mean, mel_std))
    feature_stats.append(
        list(row[-4:-1])
        + [int(pitch_shifted), int(speed_changed)]
        + list(row[-1:])
        + list(features_row)
    )

# Create column names: Mean and Std for each Mel band
mean_names = [f"Mel_Band_{i+1}_mean" for i in range(n_mels)]
std_names = [f"Mel_Band_{i+1}_std" for i in range(n_mels)]
columns = mean_names + std_names
all_feature_names = (
    list(voice_profiles.columns[:-1])
    + ["Pitch shifted", "Speed changed", "Gender"]
    + mean_names
    + std_names
)

print(f"Number of features extracted: {len(all_feature_names) - 6}")

# Convert list to DataFrame
df = pd.DataFrame(feature_stats, columns=all_feature_names)

# Save to CSV

excel_file_path = "./outputs/combined_feats.xlsx"  # Update this to your desired output file path
df.to_excel(excel_file_path, index=False)
print(f"Feature statistics saved to {excel_file_path}")
print(df.head())

# # frame

# n_mels = 128
# feature_stats = []
# feature_names = []

# for audio_file in tqdm(audio_files):
#     pitch_shifted = 100
#     speed_changed = 100
#     short_name = Path(audio_file).parent.stem

#     if "pitch_shifted" in audio_file:
#         short_name, pitch_shifted = Path(audio_file).stem.split("_pitch_shifted_")

#     if "speed_changed" in audio_file:
#         short_name, speed_changed = Path(audio_file).stem.split("_speed_changed_")

#     row = voice_profiles.loc[voice_profiles["short_name"] == short_name].values[0]

#     _, _, mel_spectrogram = extract_mel_features(audio_file)

#     # print(mel_spectrogram.shape)
#     feature_names = [
#         "locale",
#         "local_name",
#         "short_name",
#         "Pitch shifted",
#         "Speed changed",
#         "Gender",
#         "Frame",
#     ]
#     n_frames = mel_spectrogram.shape[1]
#     feature_names += [f"Mel_{i+1}" for i in range(mel_spectrogram.shape[0])]
#     for i in range(n_frames):
#         features_row = mel_spectrogram[:, i]

#         if not np.any(features_row):
#             continue

#         feature_stats.append(
#             list(row[:-1])
#             + [int(pitch_shifted), int(speed_changed)]
#             + list(row[-1:])
#             + [i]
#             + list(features_row)
#         )


# # # Convert list to DataFrame
# df = pd.DataFrame(feature_stats, columns=feature_names)

# # # Save to CSV

# excel_file_path = (
#     "./outputs/output_mel_frame_feature_stats.xlsx"  # Update this to your desired output file path
# )
# df.to_excel(excel_file_path, index=False)
# print(f"Feature statistics saved to {excel_file_path}")
# print(df.head())
