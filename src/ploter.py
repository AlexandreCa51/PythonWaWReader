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
from scipy.signal import argrelextrema

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
    plt.grid()
    return plt

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
    plt.grid()
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

def plot_signal_with_first_peak(signal, sr):
    # Trouver les indices des pics dans le signal
    plt.figure(figsize=(10, 4))

    # Tracer les deux premières secondes avec une échelle de temps très détaillée
    start_time = 0
    end_time = 0.02  
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    time = np.linspace(start_time, end_time, end_sample - start_sample)
    plt.plot(time, signal[start_sample:end_sample])

    # Ajouter une grille très fine
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Ajouter des labels et un titre
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.title('Premières milisecondes du signal audio')
    return plt