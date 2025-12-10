# ======================================================
# 🎤 VOICE EVALUATION & TRANSCRIPTION SYSTEM (English Fixed)
# ======================================================

import librosa
import numpy as np
import whisper
import jiwer
import parselmouth
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
import os

# ======================================================
# Weighted Accuracy Calculation
# ======================================================
def calculate_accuracy(gender, pitch_avg, pitch_std, volume, speech_rate, wer):
    score = 0
    total_weight = 100

    weights = {
        "pitch": 25,
        "modulation": 20,
        "volume": 15,
        "speech_rate": 20,
        "clarity": 20
    }

    # Pitch check (gender-based ranges)
    if gender == "Male" and 85 <= pitch_avg <= 180:
        score += weights["pitch"]
    elif gender == "Female" and 165 <= pitch_avg <= 255:
        score += weights["pitch"]

    # Pitch variation (modulation)
    if 20 <= pitch_std <= 80:
        score += weights["modulation"]

    # Volume check (dB range)
    if -30 <= volume <= -10:
        score += weights["volume"]

    # Speech rate check (words per minute)
    if 100 <= speech_rate <= 160:
        score += weights["speech_rate"]

    # Clarity check (WER)
    if wer <= 15:
        score += weights["clarity"]
    elif wer <= 30:
        score += weights["clarity"] * 0.5

    return round((score / total_weight) * 100, 2)


# ======================================================
# Voice Evaluation Function
# ======================================================
def evaluate_voice(audio_file):
    # 1️⃣ Load Audio + Noise Reduction
    y, sr = librosa.load(audio_file, sr=None)
    y = nr.reduce_noise(y=y, sr=sr)

    # 2️⃣ Normalize volume
    y = librosa.util.normalize(y)

    # 3️⃣ Pitch Detection using Parselmouth
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]

    pitch_avg = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # Gender detection
    gender = "Male" if pitch_avg < 165 else "Female"

    # 4️⃣ Volume Calculation (RMS in dB)
    rms = np.sqrt(np.mean(y ** 2))
    volume_db = 20 * np.log10(rms + 1e-6)

    # 5️⃣ Whisper Transcription (Force English)
    print("⏳ Loading Whisper model... (this may take a moment)")
    model = whisper.load_model("small")
    print("✅ Model loaded. Transcribing (forced English)...")
    result = model.transcribe(audio_file, fp16=False, language="en")  # ✅ Force English
    transcript = result["text"].strip()

    # 6️⃣ Clarity (WER) - Self comparison
    reference = transcript
    hypothesis = transcript
    wer = jiwer.wer(reference, hypothesis) * 100

    # 7️⃣ Speech Rate (Words per Minute)
    words = transcript.split()
    duration_sec = librosa.get_duration(y=y, sr=sr)
    speech_rate = (len(words) / (duration_sec / 60)) if duration_sec > 0 else 0

    # 8️⃣ Weighted Accuracy
    accuracy = calculate_accuracy(gender, pitch_avg, pitch_std, volume_db, speech_rate, wer)

    # ======================================================
    # Report Output
    # ======================================================
    print("\n🔍 --- VOICE EVALUATION REPORT ---")
    print(f"File:                {os.path.basename(audio_file)}")
    print(f"Detected gender:     {gender}")
    print(f"Pitch avg (Hz):      {pitch_avg:.2f}")
    print(f"Pitch std (Hz):      {pitch_std:.2f}")
    print(f"Volume (dB):         {volume_db:.2f}")
    print(f"Speech rate (WPM):   {speech_rate:.2f}")
    print(f"WER (%):             {wer:.2f}")
    print(f"Estimated accuracy:  {accuracy}%")
    print(f"Transcript:          {transcript}")

    # ======================================================
    # Visualization
    # ======================================================
    plt.figure(figsize=(12, 8))

    # Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title("Speech Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Volume (RMS)
    plt.subplot(3, 1, 2)
    rms_frame = librosa.feature.rms(y=y)[0]
    frames = range(len(rms_frame))
    times = librosa.frames_to_time(frames, sr=sr)
    plt.plot(times, rms_frame, color="orange")
    plt.title("Volume (RMS)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")

    # Pitch Contour
    plt.subplot(3, 1, 3)
    pitch_values_plot = pitch_values if len(pitch_values) > 0 else [0]
    plt.plot(pitch_values_plot, color="green")
    plt.title("Pitch Contour")
    plt.xlabel("Frame Index")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


# ======================================================
# Run (Auto Pick Latest Audio File)
# ======================================================
if __name__ == "__main__":
    RECORDINGS_DIR = os.path.join("media", "recordings")

    if not os.path.exists(RECORDINGS_DIR):
        print(f"⚠️ Directory not found: {RECORDINGS_DIR}")
        exit()

    wav_files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav")]
    if not wav_files:
        print("❌ No .wav audio files found in the recordings folder.")
        exit()

    wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(RECORDINGS_DIR, x)))
    recent_audio = os.path.join(RECORDINGS_DIR, wav_files[-1])

    print(f"🎧 Analyzing most recent audio file: {recent_audio}")
    evaluate_voice(recent_audio)
