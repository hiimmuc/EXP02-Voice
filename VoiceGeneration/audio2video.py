import glob
import os
from pathlib import Path

audio_filepaths = glob.glob(
    "./generated/characters_wav_files_44100Hz/generated/**/*.wav", recursive=True
)
img_path = "./generated/sound.gif"

# for audio_filepath in audio_filepaths:
#     audio_filepath = Path(audio_filepath)
#     char_name = audio_filepath.parent.stem
#     video_filepath = audio_filepath.parent.parent / "final" / f"{char_name}.mp4"
#     if video_filepath.exists():
#         os.system(f"rm {video_filepath}")

#     os.system(
#         f'ffmpeg -hide_banner -i {audio_filepath} -ignore_loop 0 -i {img_path} -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -shortest -strict -2 -c:v libx264 -threads 4 -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_filepath}'
#     )
print("Total:", len(audio_filepaths))
for audio_filepath in audio_filepaths:
    saving_path = str(audio_filepath).replace("characters_wav_files_44100Hz", "videos")
    saving_path = Path(saving_path)
    fname = saving_path.stem
    gender = saving_path.parent.parent.stem
    try:
        lang, local, char_name, content = str(fname).split("-")
        video_filepath = (
            saving_path.parent.parent.parent / content / gender / f"{lang}-{local}-{char_name}.mp4"
        )
        os.makedirs(video_filepath.parent, exist_ok=True)
        if video_filepath.exists():
            os.system(f"rm {video_filepath}")

        os.system(
            f'ffmpeg -hide_banner -i {audio_filepath} -ignore_loop 0 -i {img_path} -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -shortest -strict -2 -c:v libx264 -threads 4 -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_filepath}'
        )
        print(audio_filepath, "->", video_filepath)
    except Exception as e:
        print("Error:", repr(e), str(fname).split("-"))
        pass
