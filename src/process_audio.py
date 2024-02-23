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

from load_audio import load_audio
from calculate import *

def process_audio(audio_file):
    signal, sr = load_audio(audio_file)
    spl = calculate_spl(signal, sr)
    psd = calculate_psd(signal)
    return signal, sr, spl, psd
