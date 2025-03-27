"""
Feature extraction from audio files using pyAudioAnalysis library
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
from tqdm.auto import tqdm

print("Audio Analysis with Python")


audio_files = glob.glob("./wav_files/*.wav", recursive=True) + glob.glob(
    "./wav_files_modified/*.wav", recursive=True
)
print(f"Number of audio files: {len(audio_files)}")

voice_profiles = pd.read_csv("./outputs/voices.csv")  # Load the voice profiles
# get only en- locale voices
voice_profiles = voice_profiles[voice_profiles["locale"].str.contains("en-")]
# drop voice_type and style_list columns
voice_profiles.drop(columns=["voice_type", "style_list"], inplace=True)
# print(voice_profiles.head())
# ["locale", "local_name", "short_name", "gender"]

feature_stats = []
feature_names = []

for audio_file in tqdm(audio_files):
    # Read audio file
    [Fs, x] = audioBasicIO.read_audio_file(audio_file)
    #
    pitch_shifted = 0
    speed_changed = 0
    short_name = Path(audio_file).stem

    if "pitch_shifted" in audio_file:
        short_name, pitch_shifted = Path(audio_file).stem.split("_pitch_shifted_")

    if "speed_changed" in audio_file:
        short_name, speed_changed = Path(audio_file).stem.split("_speed_changed_")

    row = voice_profiles.loc[voice_profiles["short_name"] == short_name].values[0]

    # Extract short-term features
    F, feature_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

    # Compute mean and std for each feature across all frames
    feature_means = np.mean(F, axis=1)
    feature_stds = np.std(F, axis=1)

    # Prepare a single row with mean and std for current file
    features_row = np.concatenate((feature_means, feature_stds))
    feature_stats.append(
        list(row[-4:-1]) + [pitch_shifted, speed_changed] + list(row[-1:]) + list(features_row)
    )

# NOTE Creating a DataFrame
# Add 'audio_id' to feature names for mean and std
mean_names = [f"{name}_mean" for name in feature_names]
std_names = [f"{name}_std" for name in feature_names]
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

excel_file_path = (
    "./outputs/output_feature_stats.xlsx"  # Update this to your desired output file path
)
df.to_excel(excel_file_path, index=False)
print(f"Feature statistics saved to {excel_file_path}")
# NOTE: Mel spectrogram extraction
