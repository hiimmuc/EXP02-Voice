# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import json
import os
import wave
from pprint import pprint

import azure.cognitiveservices.speech as speechsdk
import numpy as np
import pandas as pd
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm

# cspell: ignore cognitiveservices frombuffer speechsdk wavfile

sentence_lists = json.load(open("sentences.json", "r"))
pprint(sentence_lists)
# combine 5 sentences into 1
# sample_sentence = ""
# for per, contents in range(sentence_lists.items()):
#     for intent, sentence in contents.items():
#         sample_sentence += sentence + ". "


def text2speech(text, voice_name, speech_config, save_path, sample_rate=44100):
    # Write the raw audio data to a temporary file
    temp_file = "temp.wav"
    # Initialize the speech synthesizer
    speech_config.speech_synthesis_voice_name = voice_name
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True, filename=temp_file)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    # Synthesize speech to a memory stream
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesis succeeded.")

        # Get the audio data as bytes
        audio_data = result.audio_data

        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Resample the audio to 44100 Hz
        with wave.open(temp_file, "rb") as original_wave:
            params = original_wave.getparams()
            original_sample_rate = params.framerate
            original_audio = original_wave.readframes(params.nframes)

        # Convert raw audio to numpy array
        audio_array = np.frombuffer(original_audio, dtype=np.int16)

        # Resample if needed
        if original_sample_rate != sample_rate:
            import resampy

            audio_array_resampled = resampy.resample(
                audio_array, original_sample_rate, sample_rate
            )
        else:
            audio_array_resampled = audio_array

        # Save the resampled audio to the output file
        write_wav(save_path, sample_rate, audio_array_resampled.astype(np.int16))

        print(f"Audio saved to {save_path} with sample rate 44100 Hz.")
    else:
        print(f"Speech synthesis failed: {result.reason}")


if __name__ == "__main__":
    # configurations
    save_dir = r"./generated/eHMI/"
    os.makedirs(save_dir, exist_ok=True)

    # #
    os.makedirs("outputs", exist_ok=True)

    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_key = "10905ed2d1fd4b41a7210af5f15c7fa0"
    service_region = "japaneast"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Create your client
    client = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Request the list of available voices
    voices_result = client.get_voices_async().get()

    # data frame with the list of voices
    # columns = ['locale', 'local_name', 'short_name', 'gender', 'voice_type', 'style_list  ']
    voices_df = pd.DataFrame(
        [
            [
                v.locale,
                v.local_name,
                v.short_name,
                0 if str(v.gender) == "SynthesisVoiceGender.Female" else 1,
                v.voice_type,
                v.style_list,
            ]
            for v in voices_result.voices
        ],
        columns=["locale", "local_name", "short_name", "gender", "voice_type", "style_list"],
    )
    voices_df.to_csv("outputs/voices.csv", index=False)
    voice_df = pd.read_csv("outputs/voices.csv")

    # df to csv
    try:
        for i, row in tqdm(voices_df.iterrows(), total=len(voices_df)):
            voice_name = str(row["short_name"])
            gender = str(row["gender"])
            if (
                ("en-" not in row["locale"])
                or (":" in voice_name)
                or ("Multilingual" in voice_name)
            ):
                continue

            ##NOTE: Generate single audio file

            # os.makedirs(os.path.join(save_dir, f"{voice_name}"), exist_ok=True)
            # audio_file = os.path.join(save_dir, f"{voice_name}/{voice_name}-combined.wav")
            # text2speech(
            #     text=sample_sentence,
            #     voice_name=voice_name,
            #     speech_config=speech_config,
            #     save_path=audio_file,
            # )

            ##NOTE: Personality
            # for personality, sentence in sentence_list.items():
            #     os.makedirs(os.path.join(save_dir, f"{gender}/{voice_name}"), exist_ok=True)
            #     audio_file_personality = os.path.join(
            #         save_dir, f"{gender}/{voice_name}/{voice_name}-{personality}.wav"
            #     )
            #     text2speech(
            #         text=sentence,
            #         voice_name=voice_name,
            #         speech_config=speech_config,
            #         save_path=audio_file_personality,
            #     )

            ##NOTE: Intents
            for intent, text in sentence_lists["CON"].items():
                os.makedirs(os.path.join(save_dir, f"{voice_name}"), exist_ok=True)
                audio_file_intent = os.path.join(save_dir, f"{voice_name}/{intent}.wav")
                text2speech(
                    text=text,
                    voice_name=voice_name,
                    speech_config=speech_config,
                    save_path=audio_file_intent,
                )

    except Exception as e:
        print(repr(e))
