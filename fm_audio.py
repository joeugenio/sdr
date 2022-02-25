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

# IQ samples
iq_samples = np.fromfile('./fm_rds_250k_1Msamples.iq', dtype=np.complex64)

# Demodulation
demod = np.diff(np.unwrap(np.angle(iq_samples)))

# De-emphasis filter
# low pass RC analog filter
# H(s) = 1/(RC*s + 1)
RC = 75e-6   # time constant
b = [1]      # numerator of the analog filter transfer function
a = [RC, 1]  # denominator of the analog filter transfer function

# transform the analog filter (s-domain) into a digital filter (z-domain)
# bilinear transform
bz, az = signal.bilinear(b, a, fs=FS)

# filtering
deemp = signal.lfilter(bz,az,demod)

# ---- MONO AUDIO ----
# decimate filter to get mono audio
ADF = 6 # audio decimate factor
ASR = FS/ADF # audio sample rate
mono = signal.decimate(deemp, ADF, ftype='fir')
mono /= mono.std()     # normalizes

# ---- STEREO AUDIO ----
# bandpass filtering for the 19kHz pilot tone
NTAPS = 101                      # number of taps
cutoff_pt = [18.8e3, 19.2e3]     # filter cut off
b_pt = signal.firwin(NTAPS, cutoff_pt, fs=FS, pass_zero='bandpass')
pt = signal.lfilter(b_pt, 1, demod) # get pilot tone
pt /= pt.std()             # normalizes

# bandpass filtering for the 38kHz L-R stereo audio
cutoff_pt = [22.9e3, 53.1e3]          # filter cut off
b_pt = signal.firwin(NTAPS, cutoff_pt, fs=FS, pass_zero='bandpass')
fil_lr = signal.lfilter(b_pt, 1, deemp)  # get L-R filtered

# AM coherent demodulation of L-R audio
demod_lr = fil_lr*pt*pt # shift by 38kHz

# decimate filter to get stereo L-R audio 
st_lr = signal.decimate(demod_lr, ADF, ftype='fir')
st_lr /= st_lr.std()     # normalizes

# separate stereo channels
st_l = mono + st_lr
st_r = mono - st_lr

# Interleaving L and R channels
stereo = np.stack((st_l, st_r)).reshape((-1,), order='F')
stereo /= stereo.std()   # normalizes

# ---- PCM mono audio ----
# scales to int16 range: -32768 to 32767
VF = 0.5   # volume factor 50%
mono_pcm = mono * (VF*np.iinfo(np.int16).max / abs(mono).max())
# plays mono audio on aplay
AUDIOFILE = 'mono.raw'
mono_pcm.astype("int16").tofile(AUDIOFILE)
system('aplay {} -r{} -f S16_LE -c 1 -t raw'.format(AUDIOFILE, int(ASR)))

# ---- PCM stereo audio ----
# scales to int16 range: -32768 to 32767
stereo_pcm = stereo * (VF*np.iinfo(np.int16).max / abs(stereo).max())
# plays stereo audio on aplay
AUDIOFILE = 'stereo.raw'
stereo_pcm.astype("int16").tofile(AUDIOFILE)
system('aplay {} -r{} -f S16_LE -c 2 -t raw'.format(AUDIOFILE, int(ASR)))

# ---- Plots ----
import matplotlib.pyplot as plt
N = 1024      # FFT size (for plotting purposes only)

plt.figure(figsize=(16,12))
plt.subplot(221)
m, _, _ = plt.magnitude_spectrum(demod[:N], FS, 0, scale='dB')
mdb = 20*np.log10(m)
plt.vlines(19e3, mdb.min(), mdb.max(), 'r', ls='--')     # 19kHz (pilot tone)
plt.vlines(19e3*2, mdb.min(), mdb.max(), 'r', ls='--')   # 38kHz (stereo audio)
plt.title('After demodulation')

# after deemphasis
plt.subplot(222)
m, _, _ =plt.magnitude_spectrum(deemp[:N], FS, 0, scale='dB')
mdb = 20*np.log10(m)
plt.vlines(19e3, mdb.min(), mdb.max(), 'r', ls='--')     # 19kHz (pilot tone)
plt.vlines(19e3*2, mdb.min(), mdb.max(), 'r', ls='--')   # 38kHz (stereo audio)
plt.title('After de-emphasis')

# pilot tone
plt.subplot(223)
t = np.linspace(0, pt.size/FS, pt.size)
plt.plot(t[:300], pt[:300])
plt.title('Pilot Tone 19kHz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# de-emphasis frequency response
# compute and plot frequency response of the digital filter
plt.subplot(224)
wz, hz = signal.freqz(bz, az, fs=FS)
plt.plot(wz, 20*np.log10(abs(hz)))
plt.title('De-emphasis Frequency response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.show()