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

from save_figure import save_figure
from load_audio import load_audio
from ploter import *
from calculate import *
from process_audio import *
from create_result_directory import create_result_directory


def main():
    laser = True #on etudier un tire du laser libs 
    audio_file = 'bellegarde10.wav'
    audio_file = os.path.join('../data',audio_file)  # Chemin relatif du fichier audio
    signal, sr, spl, psd = process_audio(audio_file)
    
    # Création du répertoire Resultats avec le nom de l'audio
    result_dir = create_result_directory(audio_file)
    if laser == True:
        plt_time_to_peak = plot_signal_with_first_peak(signal, sr)
        save_figure(plt_time_to_peak, result_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_time_to_pick.png")
        plt_time_to_peak.show()
        
    plt_spl = plot_spl(spl, sr)
    save_figure(plt_spl, result_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_Lp.png")
    plt_spl.show()

    plt_octave = plot_1_3_octave_spectrum(librosa.fft_frequencies(sr=sr, n_fft=psd.shape[0]), np.mean(psd, axis=1), sr)
    save_figure(plt_octave, result_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_spectre_octave.png")
    plt_octave.show()
    
    plt_psd = plot_psd(psd, sr)
    save_figure(plt_psd, result_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_PSD.png")
    plt_psd.show()
    
    plt_sgram = plot_spectrogram(signal, sr)
    save_figure(plt_sgram, result_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_Sgram.png")
    plt_sgram.show()

if __name__ == "__main__":
    main()
