# load and combine data
import glob
import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

eHMI_mapping_df = pd.read_excel("./data/Result-Raw.xlsx", sheet_name="eHMI Mapping")
participants_df = pd.read_excel("./data/Result-Raw.xlsx", sheet_name="Participants")
subjective_results_df = pd.read_excel("./data/Result-Raw.xlsx", sheet_name="Subjective")
eHMI_personality_results_df = pd.read_excel(
    "./data/Result-Raw.xlsx", sheet_name="eHMI personality"
)
eHMI_quality_results_df = pd.read_excel("./data/Result-Raw.xlsx", sheet_name="eHMI quality")
# only keep relevant columns
eHMI_personality_results_df = eHMI_personality_results_df[
    ["Participant ID", "Condition", "BFI O", "BFI C", "BFI E", "BFI A", "BFI N", "Overall"]
]
participants_df = participants_df[
    [
        "Participant ID",
        "Student ID",
        "Name",
        "Gender",
        "Nationality",
        "Condition 1",
        "Condition 2",
        "Condition 3",
        "Condition 4",
        "Condition 5",
        "Condition 6",
        "TIPI 1 O",
        "TIPI 1 C",
        "TIPI 1 E",
        "TIPI 1 A",
        "TIPI 1 N",
    ]
]

# Merge the three dataframes on 'Participant ID' and 'Condition'
# First check if column names exist in the dataframes
print("subjective_results_df columns:", subjective_results_df.columns.tolist())
print("eHMI_personality_results_df columns:", eHMI_personality_results_df.columns.tolist())
print("eHMI_quality_results_df columns:", eHMI_quality_results_df.columns.tolist())

# Assume the column names are 'Participant ID' and 'condition'
# Standardize column names if needed
for df in [subjective_results_df, eHMI_personality_results_df, eHMI_quality_results_df]:
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Merge dataframes
data_df = pd.merge(
    subjective_results_df,
    eHMI_personality_results_df,
    on=["Participant ID", "Condition"],
    how="inner",
)

data_df = pd.merge(
    data_df, eHMI_quality_results_df, on=["Participant ID", "Condition"], how="inner"
)

# Check the merged dataframe
print("data_df shape:", data_df.shape)
print("Unique participant IDs:", data_df["Participant ID"].nunique())
print("Unique conditions:", data_df["Condition"].unique())


# Step 1: First merge participants_df with data_df on Participant ID
participant_columns = [
    "Participant ID",
    "Student ID",
    "Name",
    "Gender",
    "Nationality",
    "TIPI 1 O",
    "TIPI 1 C",
    "TIPI 1 E",
    "TIPI 1 A",
    "TIPI 1 N",
]

data_with_participants = pd.merge(
    data_df, participants_df[participant_columns], on="Participant ID", how="left"
)

# Step 2: Create a mapping dictionary to convert Condition number to actual condition ID
# Each participant has their own condition ordering
condition_map = {}
for _, row in participants_df.iterrows():
    participant_id = row["Participant ID"]
    for i in range(1, 7):  # Assuming there are 6 conditions
        condition_col = f"Condition {i}"
        if condition_col in row:
            condition_map[(participant_id, i)] = row[condition_col]

# Step 3: Apply the mapping to create a new column with the actual eHMI ID
data_with_participants["eHMI_ID"] = data_with_participants.apply(
    lambda row: condition_map.get((row["Participant ID"], row["Condition"]), None), axis=1
)

# Step 4: Merge with eHMI_mapping_df to get 'Type' and 'Gender of eHMI'
database_df = pd.merge(
    data_with_participants,
    eHMI_mapping_df[["eHMI ID", "Type", "Gender", "EXT", "AGR", "CON", "NEU", "OPN"]],
    left_on="eHMI_ID",
    right_on="eHMI ID",
    how="left",
)

# Rename columns for clarity
database_df = database_df.rename(
    columns={"Gender_x": "Participant Gender", "Gender_y": "eHMI Gender"}
)

# Clean up by dropping redundant columns if needed
database_df = database_df.drop("eHMI ID", axis=1, errors="ignore")

# Verify the merged dataframe
print("database_df shape:", database_df.shape)
print("Unique participant IDs:", database_df["Participant ID"].nunique())
print("Unique eHMI IDs:", database_df["eHMI_ID"].nunique())
print("Type values:", database_df["Type"].unique())

# Reorganize columns in database_df

# Identify column groups
participant_cols = [
    "Participant ID",
    "Student ID",
    "Name",
    "Participant Gender",
    "Nationality",
    "TIPI 1 O",
    "TIPI 1 C",
    "TIPI 1 E",
    "TIPI 1 A",
    "TIPI 1 N",
]

condition_cols = ["Condition", "eHMI_ID", "Type", "eHMI Gender", "EXT", "AGR", "CON", "NEU", "OPN"]

# Get all other columns that aren't in either group
all_cols = database_df.columns.tolist()
other_cols = [col for col in all_cols if col not in participant_cols and col not in condition_cols]

# Create new column order
new_col_order = participant_cols + condition_cols + other_cols

# Reorder the dataframe columns
database_df = database_df[new_col_order]

# Display the first few rows to verify the new order
print("Reorganized database_df columns:")
print(database_df.columns.tolist())
print(database_df.head())

# Export to CSV if needed
database_df.to_csv("./output/processed_data/database_df.csv", index=False)
