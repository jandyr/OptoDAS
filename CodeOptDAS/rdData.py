#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________System Utilities__________/
import obspy
import os
import h5py
import numpy as np
import random
import math
import pprint
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt
from scipy import constants 
#\__________Local functions__________/
#           +-> path[0] is reserved for script path
import utils as u
import plot as p
#\_____________________________________________/
#
# -------------- Main Program --------------
""" ΣᵢCᵢ → ΣX λᵢ ΣᵢCᵢ & ΣᵢXᵢ  ≲ ⪆
1) Units 
strain, strain/s, strain/s
rad/m , rad/s   , rad/(s*m)
2) Best dates
     Date      Sampling Rate Channel Spacing Gauge Length   File duration: ΔT=10s
 Nov 12, 2025      8 Hz           10 m           40 m       Experiment start
  Jan 6, 2026      8 Hz           10 m           40 m

3)  .
    ├── Read/stack data
    ├── Detect DAS arrays gaps
    └── Detects bad DAS channels
"""
#+───────────────────────── Load Data ──────────────────────────────────────+
#---------- Read file / Load data
data, mData = u.chFil()
#------------ Create time/distance vectors and data limits
# Data is spatially best over the first ~10 km (krauss2026distributed).
#--- Distance and wavenumber
x  = np.arange(data.shape[1]) * mData.get('dx') # Dist. vector mData.get('dimensionSizes')
X  = (x.size - 1) * mData.get('dx')                                    # Total dist.
print(f"\n>> Choose a section on the {X / 1000.:.3f}km-long FO within ≲ 1.5km")
print(f"    1) Enter initial position and length as: x(0) D, ALL in km.")
print(f"    2) x(0)    D     (optional; ─ → not provided):")
print(f"        ─      D  →  0  D  [km]")
print(f"        ─      ─  →  0 1.5 [km]")
print(f"        ─     <0  →  0 Full length")
ent = input(f'<<  3) Enter "x(0) D":')
if not ent:
     mask = (x >= 0.) & (x <= 1500.)
else:
     parts = [p.strip() for p in ent.replace(',', ' ').split() if p.strip()]
     try:
          values = [float(p) for p in parts]
     except ValueError:
          raise ValueError("x(0) D.\n")
#
     if len(values) == 2:
          values = np.array(values) * 1000.
          mask = (x >= values[0]) & (x <= values[1])
     else:
          values = float(values[0]) * 1000.
          values = values if (values > 0. and values <= X) else X
          mask = (x >= 0.) & (x <= values)
#
#------------ Filter distance along FO
x  = x[mask]                                                     # Distance vector for dx
X  = (x.size - 1) * mData.get('dx')                              # Total length of FO
print(f">> Total length of the FO is {X / 1000.:.2f}km.")
#--- Depth along FO
z = u.depth(x)
# p.pltFOdepth(x, z)
kN = .9 / (2. * mData.get('dx'))                                 # 90% Wav. number Nyquist
k0 = 1. / (2. * X)                                               # Lowest wav. number
#--- Shorten dataset accordingly
data = data[:, mask]
#
#--- Time and frequency
t = np.arange(data.shape[0]) * mData.get('dt')                   # Time vector
T = (t.size - 1) * mData.get('dt')                               # Total time window ΔT
fN = .9 / (2. * mData.get('dt'))                                 # 90% sampling Nyquist
f0 = 1. / (2. * T)                                               # Lowest frequency
#
print(f"\n>> The FO working distance is D = {X/1000.:.3f}km:")
print(f"    1) {x.size} channels {mData.get('dx'):.3f}m long and gauge length {mData["gaugeLength"]:.3f}m.")
print(f"    2) Wave number range: Δk=[{k0:.3f}, {kN:.3f}]1/m.")
#
dummy = u.hms(T)
print(f"\n>> Time window ΔT={dummy[0]}h, {dummy[1]}min, {dummy[2]:.4f}s, or ΔT={T:.4f}s.")
print(f"    1) This corresponds to {t.size} time intervals of {mData.get('dt'):.3f}s.")
print(f"    2) Allowing a working frequency range of Δf=[{f0:.4f}, {fN:.3f}]Hz\n")
#+───────────────────────── Processing Strategy ──────────────────────────────────────+
"""
(A) For the multi-span DAS of Nokia Bell Labs and from sects 2.3 & 4 of {lindsey2020broadband}:
            𝜆      
     ε = ────────
          4π𝜁nGl
where:
1) 𝜁 is a multiplicative factor dependent on known material properties. 𝜁 = 0.79, for pure silica
 glass with an average Poisson's ratio of 𝜈 = 0.16 at 𝜆 = 1550 nm.
2) n = mdata{'cable_refractiveIndexes'}; Gl = mdata{'gaugeLength'}
3) 𝜆 = 1550.12×10^−9

(B) This code concentrates rather on the OptoDAS system of Alcatel Subsea Networks.

(C) Strain rate is recorded at every channel dx, which will be averaged over the entire gauge length Gl.

(D) Strategy
    ├── Raw phase-shift → strain rate (∂ε/∂t: no low-frequency phase drift)
    ├── Bandpass filter ∂ε/∂t
    ├── Conversion Σᵢdxᵢ → Gl
    ├── Time-Domain and Frequency-Domain weighting to suppress transient spikes.
    └── Cross-correlation on Gl.
Freq ranges from {landro2022sensing} "Sensing whales, storms, ships and earthquakes using an Arctic fibre optic cable"
"""
#+───────────────────────── Phase to strain ──────────────────────────────────────+
#
print(f' Convert raw phase to strain rate '.center(70, "─"), '\n')
#--- Conversion factor for converting the 16-bit integers into physical optical phase changes.
#     Each 1 bit (or 1 Count) equals $\pi / 2^{13}$ radians.
convAD   = (np.pi*2**-13)
#--- Factor for converting two consecutive time samples to a temporal rate AND normalizing 
#     over one gauge length, the practical length unit.
rateNorm = convAD / (mData["dt"] * mData["gaugeLength"])
#--- Transform to strain rate using the sensitivity [rad/(strain*m)]
strRate  = data * rateNorm / mData["sensitivities"] 
#--- Integrate to strain structured as Time X Space: Disabled as strain is unstable.
# strain = cumulative_trapezoid(strRate, axis=0, dx=mData["dt"])
#
print(f">> {data.shape} raw phase → {strRate.shape} ∂ε/∂t.")
print(f' Processing strain rate '.center(70, "─"), '\n')
#---------- Filtering
"""
#--- Work with 90% of Nyquist. 4/8-pole Butterworth filter.
#    1) fs = acoustic sampling rate
#    2) fN = 90% of Nyquist frequency

fs = 1. / mData.get('dt')
sos = butter(4, fN, output='sos', btype='lowpass', fs= fs)
#            └──> # of poles: 4, 8
#--- Apply sosfiltfilt along the first dimension (axis=0)
strRate = sosfiltfilt(sos, strRate, axis=0)
strain  = sosfiltfilt(sos, strain, axis=0)
print(f"\n>> High pass anti-alias filter w/ corner freq at 90% of Nyquist, fc={fN:.3f}Hz.")
"""
print(f"\n>> Apply band-pass filter to ∂ε/∂t.")
print(f"    1) ΔT={T:.4f}s → Δf=[{f0:.4f}, {fN:.3f}]Hz (default).")
print(f"    2) Ambient frequency ranges:")
a = max(0.01, f0); b = fN
print(f"    2.1) Earthquakes & distant storms                → f=[{a:.2f} {b:.2f}]Hz")
a = max(0.04, f0); b = min(0.1, fN)
print(f"    2.2) Microseismics (primary & secondary; swell?")
print(f"         Gravity/infragravity waves                  → f=[{a:.2f} {b:.2f}]Hz")
a = max(1, f0); b = min(10, fN)
print(f"    2.3) T-phase                                     → f=[{a:.2f} {b:.2f}]Hz")
print(f"    2.4) Off range: whale & ships                    → f=[10 25] | [35 85]Hz")
#--- Split by comma or space and filter out empty strings
ent = input(f'<< Enter f0 (opt.) f1 (req.) in Hz (rtn=dflt):') or '0'
parts = [p.strip() for p in ent.replace(',', ' ').split() if p.strip()]
try:
     values = [float(p) for p in parts]
except ValueError:
     raise ValueError("f0, f1.\n")
#
if len(values) == 1:
     if math.isclose(values[0], 0):
          f1 = fN                                      # f0 already defined
     else:
          f1 = values[0] if values[0] < fN else fN     # f1 <= fN
else:
     f0 = values[0] if values[0] > f0 else f0          # value[0] > f0
     f1 = values[1] if values[1] < fN else fN
#
print(f"\n>> Band-pass filter to ∂ε/∂t: [{f0:.4f}, {fN:.3f}]Hz")
#---------- Band-pass filter
strRate = u.zPhBpass(strRate, mData["dt"], f0, f1, corners=4)
#+───────────────────────── Spatially resampling of strain rate to Gl ──────────────────────────────────────+
"""
Measurements are recorded at every channel with spatial sampling interval dx averaged over a
 gauge length Gl>>dx, resulting in highly oversampled spatial. The Nyquist-Shannon sampling theorem 
 requires at least 50% overlap between consecutive Gl. Nevertheless aim to 75% to 80% overlap.
"""
strRateGl, xGl = u.resDAS(strRate, x, mData["gaugeLength"], 80)
#           └──> Gauge length positions
#--- Linear bathimetry
zGl = u.depth(xGl)
#--- Strain along FO
#strainRes, _     = u.resDAS(strain,  x, mData["gaugeLength"], 80)
#+───────────────────────── Spatially resampling of strain rate to Gl ──────────────────────────────────────+
#---------- Plot
#p.splot(strain, strRate, t, mData, vm=2e-6, xdec=8, tdec=1, chn=100)
print(f"\n>> {data.shape} raw phase → {strRate.shape} ∂ε/∂t → Gl → {strRateGl.shape}.")
#
p.plotTS(strRate, xGl, zGl, t, mData, title = '∂ε(dx)/∂t')
p.plotTS(strRateGl, xGl, zGl, t, mData, title = '∂ε(Gl)/∂t')
#+───────────────────────── f-k transform of DAS strain-rate ──────────────────────────────────────+
fk, f, k = u.fkTrm(strRate, t, x, taper=0.1)
kmax = 1. / x[1]              #avoid inf
p.plot_fk(fk, f, k, fmax=f1, kmax=kmax, velocities=[20, 1500], title = 'f-k ∂ε(dx)/∂t')

fk, f, k = u.fkTrm(strRateGl, t, xGl, taper=0.1)
kmax = 1. / xGl[0]
p.plot_fk(fk, f, k, fmax=f1, kmax=kmax, velocities=[20, 1500], title = 'f-k ∂ε(Gl)/∂t')
#+───────────────────────── Strain PSD ──────────────────────────────────────+
n_fft=256; hop_length=64; win_length = 256; dummy = 1./mData.get('dt')
#---------- Choose channel
print(f"\n>> Choose a Gl position for PSD in [{xGl[0]/1000.:.1f} {xGl[-1]/1000.:.1f}]km")
print(f"    1) The default (rtn) is Gl = 0km;")
print(f"    2) Break from the loop with Gl < 0.")
while True:
    ent = input(f'<< Gl position (dflt/rtn=0):') or '0'
    ent = float(ent.rstrip()) * 1000.
    if ent < 0: break             # Break free 
    idx = np.argmin(np.abs(xGl - ent))
#--- go for it
    periodogram, pltPar = u.pGram(strRateGl, dummy, idx, n_fft=n_fft, hop_length=hop_length, win_length = win_length)
    pltPar.extend([idx, xGl[idx]/1000., f0, f1])
    #
    p.pltpGram(t, strRateGl, periodogram, pltPar, n_fft=n_fft, hop_length=hop_length)
#
#+───────────────────────── Cross-correlation ──────────────────────────────────────+
"""
1. Frequency-Domain Filtering. Apply a bandpass filter (e.g., 0.5 Hz to 5.0 Hz for ambient noise, 
    or up to 20 Hz for active sources) before cross-correlating. This avoids the $L_2$ norm causing the
    dominance of uncorrelated optical noise or transient, highly localized noise spikes.
2. Force the correlation to rely on signal phase, ignoring localized amplitude spikes by enabling the 1-bit normalization and spectral 
 whitening.
3. Stack (average) the cross-correlations over many windows—often hours or days of data. The 
    incoherent random correlations will destructively interfere and cancel out, while the coherent stationary phases 
    will constructively interfere, gradually raising the apparent coherence.
4. In DAS systems the scattered light at a specific spatial bin destructively interferes, resulting in an extremely low 
signal-to-noise ratio, causing deadk channels.
"""
#---------- compute x-cor between a single DAS channel as virtual source
#--- Factor for estimating the maximum time lag
fact = [ [0.05,0.1], [0.1,0.25], [0.25,0.5] ]
#
print(f"\n<< Enter the max lag(s):")
print(f'   ├─ Short delays and local waveform similarity.. [{fact[0][0]*T:.3f},{fact[0][1]*T:.3f}]s')
print(f'   ├─ Moderate propagation delays................. [{fact[1][0]*T:.3f},{fact[1][1]*T:.3f}]s')
print(f'   └─ Longer delays, little overlapp.............. [{fact[2][0]*T:.3f},{fact[2][1]*T:.3f}]s')
dummy = fact[2][0]*T
ent = input(f'   Enter max lag(s) (rtn={fact[2][0]*T:.3f}):') or str(fact[2][0]*T)
max_lag = float(ent.strip())
max_lag_npts = int(max_lag/mData["dt"])
print(f">> Max lag: {max_lag:.3f}s, or {max_lag_npts}points.")
#---------- Choose base channel
print(f"\n>> Position loop for a virtual source along the {X/1000.:.1f}km-long FO.")
print(f"    1) The position should be within [{xGl[0]/1000.:.1f} {xGl[-1]/1000.:.1f}]km.")
print(f"    2) A random postion is obtained with a 'rtn'.")
print(f"    3) Break from the loop with a number < 0.")
#
white=False; bit=True
while True:
     ent = input(f'<< Gl position (rtn=random):')
     if not ent:
          idxsource = random.randint(0, len(xGl) - 1)
     else:
          ent = float(ent.rstrip()) * 1000.
          if ent < 0: break             # Break free
          idxsource = np.argmin(np.abs(xGl - ent))
#--- Source
     dummy = xGl[idxsource] / 1000.
     dummy = [f"Source at {dummy:.1f}km (whit+1bit)",
              f"Source at {dummy:.1f}km (whit)",
              f"Source at {dummy:.1f}km (1bit)"
              ]
     dummy = dummy[0] if (white & bit) else (dummy[1] if white else dummy[2])
     cc = u.computeCC(strRateGl, mData["dt"], max_lag, isource=idxsource, whitening=white, one_bit=bit)
     p.plotTxCor(strRateGl, cc, max_lag_npts, x, mData["dt"], x[idxsource], z, title=dummy)
