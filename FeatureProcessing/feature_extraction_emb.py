"""
Extract embeddings of acoustic features from audio files using pyannote.audio
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pyannote.audio import Inference, Model
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


feats_len = 0
feature_stats = []
feature_names = []

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole").to(torch.device("cuda"))

for audio_file in tqdm(audio_files):
    pitch_shifted = 100
    speed_changed = 100
    short_name = Path(audio_file).stem

    if "pitch_shifted" in audio_file:
        short_name, pitch_shifted = Path(audio_file).stem.split("_pitch_shifted_")

    if "speed_changed" in audio_file:
        short_name, speed_changed = Path(audio_file).stem.split("_speed_changed_")

    row = voice_profiles.loc[voice_profiles["short_name"] == short_name].values[0]

    emb = inference(audio_file).reshape(-1)

    feats_len = len(emb)

    # Prepare a single row with mean and std for current file
    feature_stats.append(
        list(row[-4:-1]) + [int(pitch_shifted), int(speed_changed)] + list(row[-1:]) + list(emb)
    )

# Create column names: Mean and Std for each Mel band
feat_names = [f"Feats_{i+1}" for i in range(feats_len)]
all_feature_names = (
    list(voice_profiles.columns[:-1]) + ["Pitch shifted", "Speed changed", "Gender"] + feat_names
)

print(f"Number of features extracted: {len(all_feature_names) - 6}")

# Convert list to DataFrame
df = pd.DataFrame(feature_stats, columns=all_feature_names)

# Save to CSV

excel_file_path = (
    "./outputs/output_emb_feature_stats.xlsx"  # Update this to your desired output file path
)
df.to_excel(excel_file_path, index=False)
print(f"Feature statistics saved to {excel_file_path}")
print(df.head())
