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

def create_result_directory(audio_file):
    audio_name = os.path.splitext(os.path.basename(audio_file))[0]
    result_dir = os.path.join("..", "Resultats", f"Resultats_{audio_name}")
    os.makedirs(result_dir, exist_ok=False)
    return result_dir