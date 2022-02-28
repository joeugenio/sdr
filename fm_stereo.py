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
VF = .5       # volume factor


AUDIOFILE = 'stereo.raw'

# IQ samples
iq_samples = np.fromfile('./fm_rds_250k_1Msamples.iq', dtype=np.complex64)

# Demodulation
gain = FS/(2 * np.pi * MDV)
demod = gain * np.angle(iq_samples[:-1]*iq_samples.conj()[1:])

# decimate filter to get mono audio
mono = signal.decimate(demod, ADF, ftype='fir')

# bandpass filtering for the 19kHz pilot tone
NTAPS = 101                      # number of taps
cutoff_pt = [18.9e3, 19.1e3]     # filter cut off
b_pt = signal.firwin(NTAPS, cutoff_pt, fs=FS, pass_zero='bandpass')
pt = signal.lfilter(b_pt, 1, demod) # get pilot tone
pt -= pt.mean()
pt *= 10    # Compensation for reduced amplitude in transmission (10%)

# bandpass filtering for the 38kHz L-R stereo audio
cutoff_lr = [22.9e3, 53.1e3]          # filter cut off
b_lr = signal.firwin(NTAPS, cutoff_lr, fs=FS, pass_zero='bandpass')
fil_lr = signal.lfilter(b_lr, 1, demod)  # get L-R filtered

# AM coherent demodulation of L-R audio
# cos(2x) = 2cos^2(x)-1
carrier = 2*(2*pt**2 - 1)
demod_lr = fil_lr*carrier # shift by 38kHz

# decimate filter to get stereo L-R audio 
st_lr = signal.decimate(demod_lr, ADF, ftype='fir')

# De-emphasis filter H(s) = 1/(RC*s + 1)
b = [1]      # numerator of the analog filter transfer function
a = [RC, 1]  # denominator of the analog filter transfer function

# transform the analog filter (s-domain) into a digital filter (z-domain)
# via bilinear transform
bz, az = signal.bilinear(b, a, fs=FS)

# filtering
mono = signal.lfilter(bz, az, mono)
st_lr = signal.lfilter(bz, az, st_lr)

# separate stereo channels
st_l = mono + st_lr
st_r = mono - st_lr

# remove DC offset
st_l -= st_l.mean()
st_r -= st_r.mean()

# Interleaving L and R channels
stereo = np.stack((st_l, st_r)).reshape((-1,), order='F')

stereo_pcm = stereo * (VF*np.iinfo(np.int16).max / abs(stereo).max())
# plays stereo audio on aplay
stereo_pcm.astype("int16").tofile(AUDIOFILE)
system('aplay {} -r{} -f S16_LE -c 2 -t raw'.format(AUDIOFILE, int(ASR)))