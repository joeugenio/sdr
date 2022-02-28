#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FM demodulator
@author: Joel Cordeiro
"""

import numpy as np
from scipy import signal
from os import system

FS = 250e3    # sample frequency
F0 = 99.5e6   # center frequency
MDV = 75e3    # Maximum frequency deviation
RC = 75e-6    # time constant for de-emphasis
N = 1024      # FFT size (for plotting purposes only)
ADF = 6       # audio decimate factor
ASR = FS/ADF  # audio sample rate

AUDIOFILE = 'mono.raw'

# IQ samples
iq_samples = np.fromfile('./fm_rds_250k_1Msamples.iq', dtype=np.complex64)

# Demodulation
gain = FS/(2*np.pi*MDV)
demod = gain * np.angle(iq_samples[:-1]*iq_samples.conj()[1:])

# decimate filter to get mono audio
mono = signal.decimate(demod, ADF, ftype='fir')

# De-emphasis filter H(s) = 1/(RC*s + 1)
b = [1]      # numerator of the analog filter transfer function
a = [RC, 1]  # denominator of the analog filter transfer function

# transform the analog filter (s-domain) into a digital filter (z-domain)
# via bilinear transform
bz, az = signal.bilinear(b, a, fs=FS)

# filtering
mono_deemp = signal.lfilter(bz, az, mono)
# remove DC offset
mono_deemp -= mono_deemp.mean()

# scales to int16 range: -32768 to 32767
VF = 0.5   # volume factor 50%
mono_pcm = VF * mono_deemp *np.iinfo(np.int16).max

# plays mono audio on aplay
mono_pcm.astype("int16").tofile(AUDIOFILE)
system('aplay {} -r{} -f S16_LE -c 1 -t raw'.format(AUDIOFILE, int(ASR)))
