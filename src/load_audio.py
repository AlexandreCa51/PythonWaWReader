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


def load_audio(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)
    return signal, sr