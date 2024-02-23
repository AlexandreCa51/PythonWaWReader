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

def save_figure(fig, result_dir, filename):
    filepath = os.path.join(result_dir, filename)
    fig.savefig(filepath)
    print(f"Figure enregistrée sous {filepath}")