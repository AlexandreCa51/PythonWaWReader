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


def calculate_spl(signal, sr, epsilon=1e-10):
    spl = 20 * np.log10((np.abs(signal) + epsilon) / (20e-6))
    return spl

def calculate_psd(signal):
    psd = np.abs(librosa.stft(signal))**2
    return psd
