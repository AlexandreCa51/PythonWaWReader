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




def plot_signal_with_first_peak(signal, sr):
    # Calculer le temps jusqu'au premier pic
    # Trouver les indices et les amplitudes des pics dans le signal
    peaks, _ = librosa.find_peaks(signal)

    # Si aucun pic n'est trouvé, retourner None
    if len(peaks) == 0:
        time_to_peak = None
    else:
        # Calculer le temps jusqu'au premier pic (en secondes)
        time_to_peak = peaks[0] / sr

    # Tracer le signal
    plt.figure(figsize=(10, 6))

    # Tracer les deux premières secondes avec une échelle de temps très détaillée
    start_time = 0
    end_time = 2  # Deux premières secondes
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    time = np.linspace(start_time, end_time, end_sample - start_sample)
    plt.plot(time, signal[start_sample:end_sample])

    # Ajouter un marqueur pour le premier pic
    if time_to_peak is not None:
        plt.scatter(time_to_peak, signal[int(time_to_peak * sr)], color='red', label='Premier pic')

    # Ajouter des labels et un titre
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.title('Deux premières secondes du signal audio')

    # Afficher la légende si un pic a été trouvé
    if time_to_peak is not None:
        plt.legend()

    plt.show()

    return time_to_peak

