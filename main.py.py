# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:20:51 2024

@author: Ukusson
"""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def save_figure(fig, filename):
    filepath = os.path.join("Resultats", filename)
    fig.savefig(filepath)
    print(f"Figure enregistrée sous {filepath}")

def load_audio(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)
    return signal, sr

def calculate_spl(signal, sr, epsilon=1e-10):
    spl = 20 * np.log10((np.abs(signal) + epsilon) / (20e-6))
    return spl

def plot_spl(spl, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(spl)) / sr, spl)
    plt.title('Niveau de Pression Acoustique (Lp)')
    plt.xlabel('Temps (s)')
    plt.ylabel('Lp (dB)')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%0.1f'))  # Format avec une décimale
    plt.tight_layout()
    return plt

def plot_1_3_octave_spectrum(freqs, psd, sr):
    center_frequencies = np.zeros(29)
    for i in range(29):
        center_frequencies[i] = 1000 * 2 ** (i / 3)
    octave_magnitudes = np.zeros(29)
    for i in range(29):
        idx = np.where((freqs >= center_frequencies[i] / np.sqrt(2)) & (freqs <= center_frequencies[i] * np.sqrt(2)))[0]
        octave_magnitudes[i] = np.sum(psd[idx])
    plt.figure(figsize=(10, 4))
    plt.plot(center_frequencies, octave_magnitudes)
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%0.1f'))  # Format avec une décimale
    plt.title('Spectre par tiers d\'octave')
    plt.tight_layout()
    return plt

def calculate_psd(signal):
    psd = np.abs(librosa.stft(signal))**2
    return psd

def plot_psd(psd, sr):
    plt.figure(figsize=(10, 4))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=psd.shape[0])
    plt.plot(freqs[:len(freqs)//2], np.mean(psd, axis=1)[:len(freqs)//2])  # Utilisation des demi-fréquences
    plt.xscale('log')  # Utilisation d'une échelle logarithmique sur l'axe x
    plt.title('Densité Spectrale de Puissance (PSD)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('PSD')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%0.1f'))  # Format avec une décimale
    plt.tight_layout()
    return plt

def plot_spectrogram(signal, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(signal))), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    plt.xlabel('Temps (s)')
    plt.ylabel('Fréquence (Hz)')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%0.1f'))  # Format avec une décimale
    plt.tight_layout()
    return plt

def process_audio(audio_file):
    signal, sr = load_audio(audio_file)
    spl = calculate_spl(signal, sr)
    psd = calculate_psd(signal)
    return signal, sr, spl, psd

def main():
    audio_file = 'audio_heli.wav'  # Remplacez par le chemin de votre fichier audio
    signal, sr, spl, psd = process_audio(audio_file)

    plt_spl = plot_spl(spl, sr)
    save_figure(plt_spl, f"{os.path.splitext(audio_file)[0]}_Lp.png")
    plt_spl.show()

    plt_octave = plot_1_3_octave_spectrum(librosa.fft_frequencies(sr=sr, n_fft=psd.shape[0]), np.mean(psd, axis=1), sr)
    save_figure(plt_octave, f"{os.path.splitext(audio_file)[0]}_spectre_octave.png")
    plt_octave.show()

    plt_psd = plot_psd(psd, sr)
    save_figure(plt_psd, f"{os.path.splitext(audio_file)[0]}_PSD.png")
    plt_psd.show()

    plt_sgram = plot_spectrogram(signal, sr)
    save_figure(plt_sgram, f"{os.path.splitext(audio_file)[0]}_Sgram.png")
    plt_sgram.show()

if __name__ == "__main__":
    main()
