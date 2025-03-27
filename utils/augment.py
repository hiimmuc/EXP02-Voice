import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile

# ====================================================Audio Augmentation utils==================================
# Time domain


def gain_target_amplitude(sound, target_dBFS=-10):
    if target_dBFS > 0:
        return sound
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def pitch_shift(sound, changes=None, n_step=0.0, n_octave_bin=12, sr=8000):
    # shift the pitch up by half an octave (speed will increase proportionally)
    # 1 octave = 12 half steps = 12 semitones
    if changes is not None:
        octaves = changes
    else:
        octaves = n_step / n_octave_bin

    new_sample_rate = int(sound.frame_rate * (2.0**octaves))

    # keep the same samples but tell the computer they ought to be played at the
    # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
    hipitch_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_sample_rate})

    # now we just convert it to a common sample rate (44.1k - standard audio CD) to
    # make sure it works in regular audio players. Other than potentially losing audio quality (if
    # you set it too low - 44.1k is plenty) this should now noticeable change how the audio sounds.
    hipitch_sound = hipitch_sound.set_frame_rate(sr)
    return hipitch_sound


def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    # slow_sound = speed_change(sound, 0.75)
    # fast_sound = speed_change(sound, 2.0)

    sound_with_altered_frame_rate = sound._spawn(
        sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed)}
    )
    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
