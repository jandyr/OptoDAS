#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________System Utilities__________/
import h5py
import requests
from bs4 import BeautifulSoup
import obspy
import tempfile
import os
import h5py
import numpy as np
import random
import torch
from torch import fft
#import torch.fft as fft
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm                   #progress bar
#\__________Scripts List__________/
"""
Script          Description


"""
#\_____________________________________/
#
#\__________Scripts __________/
#+────────────────────────────────── fkTrm ─────────────────────────────────────────────+
"""
f-k (frequency-wavenumber) transform of DAS strain-rate data using PyTorch.

    Parameters
    ----------
    taper    : fraction of each axis tapered with a Tukey window (0 = no taper)
    pad_pow2 : zero-pad both axes to the next power of two (faster FFT, smoother spectrum)
    device   : 'cuda', 'cpu', ... (default: cuda if available)

Input
-----
data : np.ndarray, shape (ntime, ndistance)   strain rate
t    : np.ndarray, shape (ntime,)             time vector [s]
x    : np.ndarray, shape (ndistance,)         distance along the fiber [m]

    Returns
    -------
    fk : torch.Tensor (complex), shape (nf, nk)  -- f along dim 0, k along dim 1
    f  : torch.Tensor, shape (nf,)   frequency [Hz],  f >= 0
    k  : torch.Tensor, shape (nk,)   wavenumber [cycles/m], increasing (centered on 0)

Convention
----------
Waves of the form cos(k*x - w*t) (propagating toward +x) appear at
f > 0, k > 0. Only f >= 0 is returned, since the input is real.
k is in cycles/m (multiply by 2*pi for rad/m).
"""
def fkTrm(data, t, x, taper=0.1, pad_pow2=True, device=None, dtype=torch.float32):
    """Local function
    """
    def _tukey(n, alpha, device, dtype):
        """Tukey (tapered cosine) window; alpha = fraction of the window that is tapered."""
        w = torch.ones(n, device=device, dtype=dtype)
        width = int(alpha * (n - 1) / 2)
        if width > 0:
            ramp = 0.5 * (1 - torch.cos(torch.pi * torch.arange(width, device=device, dtype=dtype) / width))
            w[:width] = ramp
            w[-width:] = ramp.flip(0)
        return w
#+──────────────── main ────────────────────+
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = float(np.median(np.diff(t)))
    dx = float(np.median(np.diff(x)))

    d = torch.as_tensor(np.ascontiguousarray(data), dtype=dtype, device=device)
    nt, nx = d.shape

    # Pre-processing: remove the mean of each channel, then taper both axes
    d = d - d.mean(dim=0, keepdim=True)
    if taper > 0:
        d = d * _tukey(nt, taper, device, dtype)[:, None]
        d = d * _tukey(nx, taper, device, dtype)[None, :]

    nfft_t = 1 << (nt - 1).bit_length() if pad_pow2 else nt
    nfft_x = 1 << (nx - 1).bit_length() if pad_pow2 else nx

    # Time axis: real FFT (keeps f >= 0 only).
    S = torch.fft.rfft(d, n=nfft_t, dim=0)
    # Space axis: kernel exp(+i 2 pi k x) (unnormalised inverse FFT) so that
    # +x propagating waves map to k > 0 when f > 0.
    S = torch.fft.ifft(S, n=nfft_x, dim=1, norm="forward")
    S = torch.fft.fftshift(S, dim=1)  # put k = 0 in the middle

    # Scale as an approximation of the continuous Fourier transform
    S = S * (dt * dx)

    f = torch.fft.rfftfreq(nfft_t, d=dt).to(device)
    k = torch.fft.fftshift(torch.fft.fftfreq(nfft_x, d=dx)).to(device)
    return S, f, k
#+────────────────────────────────── zPhBpass ─────────────────────────────────────────────+
"""
Detects and interpolates bad DAS channels based on RMS energy. Uses conversions
A = torch.from_numpy(data[:, isource])
B = A.numpy()
Args:
    data: torch.Tensor of shape (n_channels, npts)
    mad_multiplier: float, how many MADs away from the median is considered bad.
                    (5.0 is a standard conservative threshold).
                    
Returns:
    cleaned_data: torch.Tensor with bad channels spatially interpolated.
"""
def maskBadCh(data0, mad_multiplier=5.0):
#--- # of channels
    numCh = data0.shape[1]
#--- Converts to torch. Torch expects a 2-D tensor of shape [batch, time]
    data = torch.from_numpy(data0).float().T
#---------- 1. Calculate the RMS energy of each channel over the time window
    rms = torch.sqrt(torch.mean(data**2, dim=-1))
#---------- 2. Compute Median Absolute Deviation (MAD) for robust outlier detection
    median_rms = torch.median(rms)
    mad = torch.median(torch.abs(rms - median_rms))
#---------- 3. Define acceptable energy bounds
    lower_bound = median_rms - mad_multiplier * mad
    upper_bound = median_rms + mad_multiplier * mad
#---------- 4. Create boolean masks
    is_bad = (rms < lower_bound) | (rms > upper_bound)
    is_good = ~is_bad
    
    bad_indices = torch.nonzero(is_bad).squeeze(-1)
    good_indices = torch.nonzero(is_good).squeeze(-1)
    
#    print(f"Detected and fixing {len(bad_indices)} fading/noisy channels.")
    fadCh = [len(bad_indices), len(bad_indices)/float(numCh)]
    
    if len(good_indices) == 0:
        raise Exception(f"All channels flagged as bad. Check your data.")
#
    cleaned_data = data.clone()
#---------- 5. Spatially interpolate bad channels
    for bad_idx in bad_indices:
#--- Find closest good channels to the left and right
        left_mask = good_indices < bad_idx
        right_mask = good_indices > bad_idx
#--- Edge case: Bad channel at the very start of the array
        if not left_mask.any():
            right_idx = good_indices[right_mask][0]
            cleaned_data[bad_idx] = data[right_idx]
#--- Edge case: Bad channel at the very end of the array
        elif not right_mask.any():
            left_idx = good_indices[left_mask][-1]
            cleaned_data[bad_idx] = data[left_idx]
#--- Standard case: Linear interpolation between nearest good channels
        else:
            left_idx = good_indices[left_mask][-1]
            right_idx = good_indices[right_mask][0]
            
            dist_left = bad_idx - left_idx
            dist_right = right_idx - bad_idx
            total_dist = dist_left + dist_right
            
            # Inverse distance weighting
            w_left = dist_right / total_dist
            w_right = dist_left / total_dist
            
            cleaned_data[bad_idx] = (data[left_idx] * w_left) + (data[right_idx] * w_right)

    return cleaned_data.numpy().T, fadCh
# -------------- End of function   ----------------------------
#
#+────────────────────────────────── pGram ─────────────────────────────────────────────+
"""
Strain rate periodogram
Parameters:
-----------
data :      numpy.ndarray
       2-D array of strain data with shape (nt, nx), where axis 0 is time and axis 1 is spatial channels.
fs:     Sapling frequency
idx:    Channel number
n_fft:      256 or 512
win_length: 256 (Matching the window size to n_fft provides good spectral resolution.
hop_length: 64  (A 75% overlap balance between capturing temporal changes smoothly,
                  avoiding excessive memory usage).
Returns:
--------
mean_periodogram : numpy.ndarray
     An average across all channels.
"""
def pGram(data, fs, idx, n_fft=256, hop_length=64, win_length = 256):
#---------- Convert to PyTorch Tensor
     data_tensor = torch.from_numpy(data).float()
#---------- Window
     window = torch.hann_window(win_length)
#---------- Compute STFT for each channel
#--- Torch stft expects a 2-D tensor of shape [batch, time]
     stft_out = torch.stft(
         data_tensor.T,
         n_fft=n_fft,
         hop_length=hop_length,
         win_length=win_length,
         window=window,
         return_complex=True
     )
#---------- Calculate power spectral density (periodogram over time)
#--- Shape: [n_channels, n_freqs, n_time_steps]
     periodogram = torch.abs(stft_out) ** 2
#--- Restrict to channel idx [n_freqs, n_time_steps]
     periodogram = periodogram[idx, :, :].numpy()
#---------- Average across all channels
#     mean_periodogram = torch.mean(periodogram, dim=0).numpy()
     time_steps = periodogram.shape[1]
#---------- Calculate time at each STFT column center
     t_stft = (np.arange(time_steps) * hop_length) / fs
     freqs = np.fft.rfftfreq(n_fft, d=1/fs)
#---------- Return numpy periodogram 
     return periodogram, [time_steps, t_stft, freqs]
# -------------- End of function   ----------------------------
#+────────────────────────────────── zPhBpass ─────────────────────────────────────────────+
"""
    Zero-phase Butterworth bandpass filtering in the frequency domain.

    Processing sequence:
        1. Demean
        2. Linear detrend
        3. Hann taper
        4. FFT bandpass (filtfilt-equivalent response)
        5. Inverse FFT

    Args:
        signal0 : ndarray, shape (n_channels, n_samples)
        dt      : sample interval [s]
        freqmin : low corner frequency [Hz]
        freqmax : high corner frequency [Hz]
        corners : Butterworth order
        taper_frac : fraction of trace tapered at each end
                     (typically 0.02-0.05)

    Returns:
        ndarray, shape (n_channels, n_samples)

"""
def zPhBpass(signal0, dt, freqmin, freqmax,
             corners=4, taper_frac=0.05):

    # ------------------------------------------------------------------
    # Convert to torch and transpose
    # (nch, nt) -> (nt, nch)
    # ------------------------------------------------------------------
    signal = torch.as_tensor(signal0, dtype=torch.float64).T

    npts = signal.shape[-1]
    device = signal.device

    # ------------------------------------------------------------------
    # 1. Demean
    # ------------------------------------------------------------------
    signal = signal - signal.mean(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # 2. Linear detrend (vectorized least-squares fit)
    # ------------------------------------------------------------------
    t = torch.linspace(-1.0, 1.0, npts,
                       device=device,
                       dtype=signal.dtype)

    t_mean = t.mean()
    t_centered = t - t_mean

    slope = (
        (signal * t_centered).sum(dim=-1, keepdim=True)
        / (t_centered**2).sum()
    )

    intercept = signal.mean(dim=-1, keepdim=True)

    trend = slope * t + intercept

    signal = signal - trend

    # ------------------------------------------------------------------
    # 3. Hann taper
    # ------------------------------------------------------------------
    if taper_frac > 0:

        ntaper = int(taper_frac * npts)

        if ntaper > 1:

            taper = torch.ones(npts,
                               device=device,
                               dtype=signal.dtype)

            hann = torch.hann_window(
                2 * ntaper,
                periodic=False,
                device=device,
                dtype=signal.dtype
            )

            taper[:ntaper] = hann[:ntaper]
            taper[-ntaper:] = hann[-ntaper:]

            signal = signal * taper

    # ------------------------------------------------------------------
    # Frequency axis
    # ------------------------------------------------------------------
    freqs = fft.rfftfreq(npts, d=dt).to(device)

    freqs_safe = freqs + 1e-12

    # ------------------------------------------------------------------
    # 4. Butterworth bandpass
    #
    # High-pass Butterworth
    # Low-pass Butterworth
    #
    # Squared to mimic forward/backward
    # filtfilt Butterworth response.
    # ------------------------------------------------------------------
    high_pass = 1.0 / (
        1.0 + (freqmin / freqs_safe) ** (2 * corners)
    )

    low_pass = 1.0 / (
        1.0 + (freqs_safe / freqmax) ** (2 * corners)
    )

    mask = (high_pass * low_pass) ** 2

    # Explicit DC removal
    mask[0] = 0.0

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------
    spec = fft.rfft(signal, dim=-1)

    # Apply filter
    spec *= mask

    # ------------------------------------------------------------------
    # 5. Inverse FFT
    # ------------------------------------------------------------------
    filtered = fft.irfft(spec, n=npts, dim=-1)

    return filtered.cpu().numpy().T
# -------------- End of function   ----------------------------
"""-------------------------------------------
def zPhBpass(signal0, dt, freqmin, freqmax, corners=4):
#---------- Transpose and transform to torch
    signal = torch.from_numpy(signal0).float().T
#(nt, nch) ─>(nch, nt)
    npts = signal.shape[-1]
#---------- Generate frequency bins for the real FFT
    freqs = fft.rfftfreq(npts, d=dt).to(signal.device)
#--- Add epsilon to prevent division by zero at the DC component (0 Hz)
    freqs_safe = freqs + 1e-10
#---------- Construct the Butterworth amplitude response (real-valued)
    high_pass = 1.0 / (1.0 + (freqmin / freqs_safe)**(2 * corners))
    low_pass  = 1.0 / (1.0 + (freqs_safe / freqmax)**(2 * corners))
#---------- Combine into a single bandpass mask and explicitly zero out DC
    mask = high_pass * low_pass
    mask[..., 0] = 0.0
#---------- 1. Forward FFT to frequency domain
    spec = fft.rfft(signal, dim=-1)
#---------- 2. Multiply by the real mask (alters amplitude, leaves phase perfectly untouched)
    spec_filtered = spec * mask
#---------- 3. Inverse FFT back to time domain
    filtered_signal = fft.irfft(spec_filtered, n=npts, dim=-1)
#
    return filtered_signal.numpy().T
--------------------------------------------"""

#+────────────────────────────────── resDAS ─────────────────────────────────────────────+
"""
Spatially resample DAS strain data in successive gauge length intervals Gl with v% overlap.

Parameters:
-----------
data : numpy.ndarray
    2D array of strain data with shape (nt, nx), where axis 0 is time and axis 1 is spatial channels.
x : numpy.ndarray
    1D array of shape (nx,) representing the positions of the sensing channels.
Gl : float
    Gauge length for resampling.
v : float
    Percentage of overlap between successive intervals (e.g., 50 for 50%).
    
Returns:
--------
resampled_data : numpy.ndarray
    The resampled strain data.
resampled_x : numpy.ndarray
    The spatial positions of the resampled channels.
"""
def resDAS(data, x, Gl, v=80):

    # Spatial resolution of raw channels
    dx = x[1] - x[0]
    
    # Number of channels that make up one gauge length Gl
    n_gl = int(round(Gl / dx))
    if n_gl < 1:
        n_gl = 1
        
    # Calculate step size in terms of number of channels based on overlap percentage v
    step = int(round(n_gl * (1 - v / 100.0)))
    if step < 1:
        step = 1
        
    nt, nx = data.shape
    
    resampled_slices = []
    resampled_positions = []
    
    # Slide window of size n_gl across the channels
    start_idx = 0
    while start_idx + n_gl <= nx:
        end_idx = start_idx + n_gl
        
        # Average the data within the gauge length interval
        interval_data = np.mean(data[:, start_idx:end_idx], axis=1)
        resampled_slices.append(interval_data)
        
        # Center position of this gauge length interval
        interval_center = np.mean(x[start_idx:end_idx])
        resampled_positions.append(interval_center)
        
        start_idx += step
        
    resampled_data = np.column_stack(resampled_slices)
    resampled_x = np.array(resampled_positions)
    
    return resampled_data, resampled_x
# -------------- End of function   ----------------------------
#---------- Aux. function
def nextpow2(x):
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))
"""def nextpow2(x):
    return 1 << (x - 1).bit_length()
"""
# -------------- End of function   ----------------------------
#+────────────────────────────────── torch_xcorr ─────────────────────────────────────────────+
"""Auxiliary function to estimate x-corr
Parameters
    
Returns
    
"""
def torch_xcorr(signal_1, signal_2, whitening=False, one_bit=False):
    if len(signal_1.shape) < 2 or len(signal_2.shape) < 2:
        print('input dimension must be ntrace*npts !')
        return 0
    else:
        # 1-bit normalization: retain only the sign of the signal
        if one_bit:
            signal_1 = torch.sign(signal_1)
            signal_2 = torch.sign(signal_2)

        signal_length = signal_1.shape[-1]
        x_cor_sig_length = signal_length * 2 - 1
        fast_length = nextpow2(x_cor_sig_length)
 
        # FFT along the last dimension
        fft_1 = fft.rfft(signal_1, n=fast_length, dim=-1)
        fft_2 = fft.rfft(signal_2, n=fast_length, dim=-1)
 
        # Spectral whitening: normalize amplitude spectrum to 1 for all frequencies
        if whitening:
            fft_1 = fft_1 / (torch.abs(fft_1) + 1e-10)
            fft_2 = fft_2 / (torch.abs(fft_2) + 1e-10)

        # Conjugate spectral multiplication
        fft_multiplied = torch.conj(fft_1) * fft_2
 
        # Back to time domain
        prelim_correlation = fft.irfft(fft_multiplied, n=fast_length, dim=-1)
 
        # Shift the zero-lag component to the center
        shifted = torch.roll(prelim_correlation, fast_length // 2, dims=-1)
        
        # Calculate exact indices to extract the x_cor_sig_length elements around the center
        center_pos = fast_length // 2
        start_idx = center_pos - (signal_length - 1)
        end_idx = start_idx + x_cor_sig_length
        
        final_result = shifted[:, start_idx:end_idx]
        
        # If whitening is applied, the cross-power spectrum is already normalized,
        # naturally bounding the time-domain result to [-1, 1].
        if whitening:
            return final_result
        else:
            # Calculate L2 norms of the (potentially 1-bit) signals for time-domain normalization
            norm_1 = torch.linalg.norm(signal_1, dim=-1, keepdim=True)
            norm_2 = torch.linalg.norm(signal_2, dim=-1, keepdim=True)
            return final_result / (norm_1 * norm_2 + 1e-10)
# -------------- End of function   ----------------------------
#+────────────────────────────────── computeCC ─────────────────────────────────────────────+
"""Function to compute X-cor between single DAS channel as virtual source.

Parameters
    data[nt, nchn]:
    isource:         Virtual source index

Returns
    
"""
def computeCC(data, dt, max_lag, isource=0, ch_buffer_in=-1, whitening=False, one_bit=False):
    # Accept data with shape [nt, nchn] 
    npts = data.shape[0]
    nch = data.shape[1]
    
    ch_buffer = nch if ch_buffer_in == -1 else ch_buffer_in
#
# Note change: int(nch / ch_buffer + 1.5) to math.ceil(nch / ch_buffer), to avoid overcount (18/10/26).
    nchunk = math.ceil(nch / ch_buffer) if ch_buffer < nch else 1
    max_lag_npts = int(max_lag / dt)
    cc_len = max_lag_npts * 2 + 1
    cc = np.zeros((nch, cc_len))

    # Source channel trace: shape [npts]
    dataDASsource = torch.from_numpy(data[:, isource]).double()          #.float()
#                 = torch.as_tensor(signal0, dtype=torch.float64).T
    first_ch = 0
    for ichunk in range(nchunk):
        second_ch = min(first_ch + ch_buffer, nch)
        num_ch_chunk = second_ch - first_ch
        dataDAS = torch.from_numpy(data[:, first_ch:second_ch].T.copy()).float()
 
        # Compute cross-correlation
        cc_all = torch_xcorr(
            dataDASsource.reshape(1, npts).repeat(dataDAS.shape[0], 1), 
            dataDAS,
            whitening=whitening,
            one_bit=one_bit
        ).numpy()
        # Zero-lag is at index (npts - 1) in cc_all
        center_idx = npts - 1
 
        # Source slice bounds on cc_all
        src_start = center_idx - max_lag_npts
        src_end = center_idx + max_lag_npts + 1
 
        # Destination bounds on cc
        dst_start = 0
        dst_end = cc_len
 
        # If left-side is out-of-bounds (src_start < 0)
        if src_start < 0:
            dst_start = -src_start
            src_start = 0
 
        # If right-side is out-of-bounds (src_end > cc_all.shape[1])
        if src_end > cc_all.shape[1]:
            dst_end = dst_end - (src_end - cc_all.shape[1])
            src_end = cc_all.shape[1]
 
        # Copy the valid overlapping region directly into pre-allocated zeros array
        if src_start < src_end and dst_start < dst_end:
            cc[first_ch:second_ch, dst_start:dst_end] = cc_all[:, src_start:src_end]
 
        first_ch += ch_buffer
    return cc
# -------------- End of function   ----------------------------
#+────────────────────────────────── checkConcat ─────────────────────────────────────────────+
"""Checks if two DAS arrays concatenate seamlessly without gaps.
Parameters
    data[n_channels, n_time]:


Returns
    
"""
def checkConcat(A, B, threshold_sigma=4.0):
#---------- Transpose matrices
    dataA = A.T
    dataB = B.T
#---------- Calculate internal sample-to-sample differences: dataA[:, t+1] - dataA[:, t]
    internal_diffs = np.abs(np.diff(dataA, axis=1))
#--- Calculate baseline statistics per channel (mean and std of the step size)
    mean_internal_step = np.mean(internal_diffs, axis=1)
    std_internal_step = np.std(internal_diffs, axis=1)
#---------- Calculate the boundary step between File A and File B
    boundary_step = np.abs(dataB[:, 0] - dataA[:, -1])
#---------- Check for anomalies
#--- Type 1: The step is unnaturally large (time gap)
    is_large_gap = boundary_step > (mean_internal_step + threshold_sigma * std_internal_step)
#--- Type 2: The step is exactly zero across all/most channels (dropped frame repeated)
    is_repeated_frame = boundary_step == 0
#--- Check if a significant percentage of channels show a gap (e.g., > 10% of the fiber); DAS channels can be noisy
    percent_large_gap = np.sum(is_large_gap) / len(is_large_gap) * 100
    percent_repeated = np.sum(is_repeated_frame) / len(is_repeated_frame) * 100
#--- Evaluate results
    if percent_large_gap > 10.0:
        return True, f" Gap: {percent_large_gap:.1f}% of channels w/ amplitude jump at the boundary w/ previous."
    elif percent_repeated > 95.0:
        return True, " Gap: Exact frame duplication detected at the boundary."
    else:
        return False, "No gap: Boundary continuity falls within normal statistical variance."
# -------------- End of function   ----------------------------
#
#---------- Aux. function to print attributes
#+────────────────────────────────── hdf5_item ─────────────────────────────────────────────+
def hdf5_item(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"  {name} (group)")
        if obj.attrs:
            print("    Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"      {attr_name}: {attr_value}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  {name} (dataset)")
#            print(f"{name}")
        print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
        if obj.attrs:
            print("    Attributes:")
# -------------- End of function   ----------------------------
#+──────────────────────────── Lambda Funcitions ────────────────────────────────────────+
#---------- Function to estimate depth for the first 1514m.
# Regression estimated with rdPosFO.py. Uses the 1st 3 FO points, which have the best linear fit.
# The regression coeff is R2 = 91.8%.
depth = lambda x: -0.008907*x - 4.094
# -------------- End of function   ----------------------------
#
#---------- Function to strip off path, folder and filename
S = lambda s: int(s[s.rfind('_') + 1 : s.rfind('.hdf5')])
#   └─> Extract time from filename, e.g., UW_DAS_south_8Hz_20260106_000024.hdf5
#
D = lambda d: d[:d.rfind('/') + 1]
#   └─> Extract directory from filename, e.g., /media/jmt/sdb/OptoDAS260106/
# -------------- End of function   ----------------------------
#---------- Function to transform seconds in hms
hms = lambda total_seconds: (
    total_seconds // 3600,
    (total_seconds % 3600) // 60,
    total_seconds % 60
)
# -------------- End of function   ----------------------------
#--- Lambda function to extract the time from the filenames
fileTime = lambda filename: int(filename.split('_')[-1].split('.hdf5')[0])
# -------------- End of function   ----------------------------
#--- Lambda function to transform hms to seconds
totSec = lambda t: t[0] if len(t) == 1 else (t[0]*60 + t[1] if len(t) == 2 else t[0]*3600 + t[1]*60 + t[2])
# Alternative mathematically equivalent approach using list padding / indexing from right
#totSec = lambda t: sum(val * factor for val, factor in zip(t[::-1], [1, 60, 3600]))
# -------------- End of function   ----------------------------
#+────────────────────────────────── chFil ─────────────────────────────────────────────+
"""Function to read files
"""
def chFil():
#---------- Aux. function to copy external file to a temp location
    def tempLoc(url, file):
        files = url + file
        HMS = hms(S(files))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5') as tmp_file:
                dummy = requests.get(files).content
                tmp_file.write(dummy)
                files = tmp_file.name
    #
        return files, HMS
# -------------- End of function   ----------------------------
#---------- Aux. function to choose files based on time
    def toSec(values):
#
        num_values = len(values)
        if num_values == 1:
            seconds = values[0]
            hours = minutes = 0
        elif num_values == 2:
            minutes, seconds = values
            hours = 0
        elif num_values == 3:
            hours, minutes, seconds = values
        else:
            raise Exception(f"i/P {values} not valid.")
#--- Total_seconds
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        return total_seconds
# -------------- End of function   ----------------------------
    print(' Read data '.center(70, "─")+'\n')
    ent = input(f'\n>> Read data files locally (0=dflt) or at OptoDAS url (!=0):') or '0'
#--- Read locally
    if ent == '0':
#        files=glob.glob('../Data/*.hdf5')
        files=glob.glob('/media/jmt/sdb/OptoDAS260106/*.hdf5')
        files.sort()
#
        print(f"\n>> {len(files)} datasets in {D(files[0])}:")
        print(f"   from {files[0]} at {fileTime(files[0])}s")
        print(f"   to   {files[-1]} at {hms(fileTime(files[-1]))}(h,m,s)")
        url =None

#--- Read at OptoDAS site
    else:
        url ="http://piweb.ooirsn.uw.edu/das25/data/OptoDAS/2026/01/06/south/8Hz/"
        print(f">> Fetching content from: {url}")
        temp_filepath = None # Initialize temp_filepath
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
#
            soup = BeautifulSoup(response.text, 'html.parser')
            files = []
            for link in soup.find_all('a'):
                href = link.get('href')
                # Filter for data files, typically with .mseed or .hdf5 extensions
                if href and (href.lower().endswith('.mseed') or href.lower().endswith('.hdf5')):
                    files.append(href)
#
            if not files:
                raise Exception(f"The path {fold} does not exist.")
                print("No data files at {url}.")
            else:
                print(f"\n>> {len(files)} Datasets:")
                print(f"   from {files[0]}")
                print(f"   to   {files[-1]} at {hms(fileTime(files[-1]))}(h,m,s)")
#
        except requests.exceptions.RequestException as e:
            print(f"Error accessing the URL: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
#
#---------- File picker
    """
(A) The gravity and  and Scholte waves are the primary and secondary microseisms, respectively.
     Those T-wave (tertiary wave) are low‐frequency acoustic waves (<40 Hz) that travel 
      ~10^3km with low attenuation in the ocean’s Sound Fixing and Ranging (SOFAR) channel. T waves
      travel in the water column at the speed of sound, ~1.48 km/s, lower than the P and S seismic 
      waves propagating in the crust.
  1) Shallow water: surface gravity waves (SGW) from wind-sea interactions dominate at T < 20s
  2) Deeper water:  gravity wave band shifts to infragravity waves (IGW), which dominate at T=[30-100]s.
  3) At all depths: Scholte waves dominate at T=[1-10]s. At ~350m Scholte waves and SGWs peaks are
      conspicuous at T=[1-10]s and T=[10-20]s.
  4) Earthquakes:   ≲30min.
    """
    print(f"\n>> N files extend the frequency range (90% of Nyquist).")
    print(f"  1)              ├────+─────> ΔT= 10s:  1 file")
    print(f"              Δf=[0.2  3.8]Hz ")
    print(f"  2)         ├─────────+─────> ΔT=3min, 18 files")
    print(f"        Δf=[0.01      3.8]Hz─> T, SGW, IGW, distant storms")
    print(f"  3)    ├──────────────+─────> ΔT=5min, 30 files")
    print(f"   Δf=[0.006           3.8]Hz  ")
    print(f"  4) ├──────────────────+─────> ΔT=30min ~1800 files")
    print(f"Δf=[0.001                3.8]Hz─> quakes")

#
    dummy = len(files)
    print(f"\n<< Enter the time span to read ({dummy} files):")
    print(f'    From T0 by ΔT (optional).')
    print(f'          └─────+─────> h min s')
    print(f'                └─────> min')
#--- Split by comma or space and filter out empty strings
    ent = input(f'   Enter file choice (rtn=0 10):') or '10'
    parts = [p.strip() for p in ent.replace(',', ' ').split() if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError("Please enter valid numeric integers.\n")
#
    if len(values)   == 1:
        ent   = values[0] * 60.                  # ΔT(min) ─> s
        values= 0.                               # t0
    elif len(values) >= 2:
        ent    = values[-1] * 60.                # ΔT(min) ─> s
        values = totSec(values[:-1])             # h m s   ─> t0
#
#---------- Get all file times
    fTime = [fileTime(x) for x in files]
    fTime = np.array(fTime)
#--- Create a boolean mask initialized to False
#        mask = np.zeros(fTime.shape, dtype=bool)
#--- Find the index of closest element
    idx_0 = np.argmin(np.abs(fTime - values))
    idx_1 = np.argmin(np.abs(fTime - (values + ent)))
#                                      h m s─>s + ΔT(min)
#--- Set the range between idx_0 and idx_1 (inclusive)
    files = files[min(idx_0, idx_1) : max(idx_0, idx_1) + 1]
#
    print(f"\n>> Selected {len(files)} files: {fTime[idx_0]} → {fTime[idx_1]}s")
#+───────────────── Load data ─────────────────────────────+
    print(f"\n>> Check the continuity between each contiguous files.")
#--- N = 5 MADs away from the median to be considered bad is a standard conservative threshold.
    print(f"\n>> Detects and interpolates bad DAS channels based on RMS energy (rtn=NO)?")
    ent = input(f'<< Detect bad chann. (rtn=NO):')
    nmbrMad = False
    if ent is not None:
        fadCh = []
        nmbrMad = 5
        print(f"\n>> Detects/interpolates bad DAS channels w/ {nmbrMad} MADs.")
#
    for idx, file in tqdm(enumerate(files)):
#
#---------- Download external file to a temporary location
        if url is not None: file, HMS = tempLoc(url, file)
#---------- Open file
        try:
            with h5py.File(file, 'r') as fp:
#
                if idx == 0:
#--- The meta data
                    mData = mkMetaData(fp, prtAttr=False)
#---------- Open Data container
                    data=[]
#---------- Fetch data
                data0 = np.array(fp['data'][()])
#--- Correct fading channels
                if nmbrMad:
                    data0, dummy = maskBadCh(data0, mad_multiplier=nmbrMad)
                    fadCh.extend(dummy)
#--- Check continuity
                if idx == 0:
                    prevDada = data0
                else:
                    isIt, mess = checkConcat(prevDada, data0)
                    if isIt:
                        print(f"\n>> Break of continuity at {hms(S(file))}(h:m:s). Break out of data loop.")
                        print(mess)
                        break
#
                prevDada = np.array(fp['data'][()])
#
#---------- Stack/Load data
                data.append(data0)
        #except requests.exceptions.RequestException as e:
        #    print(f"Error accessing the URL: {e}")
        except Exception as e:
            print(f"\n>> Error processing HDF5 data: {e}")
#--- Clean up the temporary file if it was created
        finally:
            if url is not None:
                if fp and os.path.exists(file):
                    os.remove(file)
#
#---------- Statistics of bad DAS channels based on RMS energy
#    if nmbrMad:
#        print(f"\n>> Downloaded {len(files)} files, stacked to a {data.shape} array")
#---------- Adjust stacked data dimensions
    data = np.array(data)
    data = data.reshape(-1, data.shape[2])
    print(f"\n>> Downloaded {len(files)} files, stacked to a {data.shape} array")
    print(f'    - Each 10s dataset is {os.path.getsize(files[0])} bytes.')
    if url is None:
        HMS = hms(S(files[0]))
        print(f'    - Recorded from {HMS[0]}h, {HMS[1]}min, {HMS[2]}s to')
        if len(files) > 1 :
            HMS = hms(S(files[-1]))
            print(f'       {HMS[0]}h, {HMS[1]}min, {HMS[2]}s.')
#
#---------- Return Data
    return data, mData
#
# -------------- End of function   ----------------------------
#
# ---------- Function to hold repeater stations ----------
"""
    Repeater locations
    <Returns>
    array([[  lats[0], lons[0]], ...
           [  lats[2], lons[2]]]])
"""
#
def repSt():
    lats = [45.14871,44.48169,44.35839,45.24241,45.48336,45.78700,45.79300,45.88690]
    lons = [-125.11577,-125.44860,-124.99590,-124.75102,-125.94169,-127.15996,-128.36265,-129.61285]
#
#------------ Return the first repetition station, which is the limit of 
#              the OptoDAS FO
#    return np.column_stack((lats[:3], lons[:3]))
    return lats[0], lons[0]
#
# -------------- End of function   ---------------------
#
# ---------- Function to fit a 3D line to a set of points ----------
"""
Fits a 3D line to a set of points and returns the sum of squared perpendicular residuals.
Uses PCA (SVD) for fitting.
Args:
    points (np.ndarray): A 2D numpy array where each row is a 3D point (x, y, z).
Returns:
    float: The sum of squared perpendicular distances from the points to the fitted line.
           Returns float('inf') if fewer than 3 points are provided.
"""
def fit_res(points):

    if points.shape[0] < 3:
        # Need at least 3 points to meaningfully assess collinearity
        return float('inf')

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform SVD on the centered data.
    # The Vh (or V transpose) matrix contains the principal components as its rows.
    # The first row (Vh[0, :]) corresponds to the first principal component,
    # which is the direction vector of the best-fit line.
    _, _, Vh = np.linalg.svd(centered_points)
    direction_vector = Vh[0, :]

    # Calculate the sum of squared perpendicular distances from points to the line.
    # The squared perpendicular distance from a point `p` to a line passing through
    # `centroid` with unit direction vector `u` is `||(p - centroid) x u||^2`.
    cross_products = np.cross(centered_points, direction_vector)
    sum_squared_residuals = np.sum(cross_products**2)

    return sum_squared_residuals
#
# -------------- End of function   ---------------------
#+───────────────────────── mkMetaData ──────────────────────────────────────+
# ---------- Function construc metadata ----------
"""
    <Parameter>
    fp: I/P file

    <Returns>
    mData
"""
def mkMetaData(fp, prtAttr=False):
#---------- List them all!!
    if prtAttr: fp.visititems(hdf5_item)
#---------- File attributes
    keys = [
            'acqRate',
            'cables',
            'cable_refractiveIndexes',
            'cable_sensDist',
            'demod_mixerCoeffX1',
            'demod_mixerCoeffX2',
            'demod_mixerCoeffY1',
            'demod_mixerCoeffY2',
            'demod_roiDec',
            'demod_roiEnd',
            'channels',
            'phiOffs',
            'computer',
            'dataScale',
            'dataType',
            'dt',
            'dx',
            'gaugeLength',
            'instrument',
            'instrumentSerial',
            'missingSamples',
            'name',
            'note',
            'dim_Names',
            'dim_max',
            'dim_size',
            'dim_unitScale',
            'dimensionSizes',
            'dimensionUnits',
            'missingSamples',
            'sensitivities',
            'sensitivityUnits',
            'sensorType',
            'sw_gcoeff_x',
            'sw_gcoeff_y',
            'instr_names',
            'instr_values',
            'sweepRate',
            'sweepFreqCtr',
            'sweepScale',
            'sweepDuration',
            'sweepName'
            ]
#---------- Fetching
    dummy = [
            fp['acqSpec/rate'][()],
            fp['cableSpec/cables'][()],
            fp['cableSpec/refractiveIndexes'][()][0],
            fp['cableSpec/sensorDistances'][()],
            fp['demodSpec/mixerCoeffX1'][()],
            fp['demodSpec/mixerCoeffX2'][()],
            fp['demodSpec/mixerCoeffY1'][()],
            fp['demodSpec/mixerCoeffY2'][()],
            fp['demodSpec/roiDec'][()],
            fp['demodSpec/roiEnd'][()],
            fp['header/phiOffs'][()],
            fp['header/channels'][()],
            fp['header/computer'][()],
            fp['header/dataScale'][()],
            fp['header/dataType'][()],
            fp['header/dt'][()],
            fp['header/dx'][()],
            fp['header/gaugeLength'][()],
            fp['header/instrument'][()],
            fp['header/instrumentSerial'][()],
            fp['header/missingSamples'][()],
            fp['header/name'][()],
            fp['header/note'][()],
            fp['header/dimensionNames'][()],
            fp['header/dimensionRanges/dimension1/max'][()],
            fp['header/dimensionRanges/dimension1/size'][()],
            fp['header/dimensionRanges/dimension1/unitScale'][()][0],
            fp['header/dimensionSizes'][()],
            fp['header/dimensionUnits'][()],
            fp['header/missingSamples'][()],
            fp['header/sensitivities'][()][0][0],
            fp['header/sensitivityUnits'][()][0],
            fp['header/sensorType'][()],
            fp['sweepSpec/gcoeff_x'][()],
            fp['sweepSpec/gcoeff_y'][()],
            fp['instrumentOptions/names'][()],
            fp['instrumentOptions/values'][()],
            fp['sweepSpec/rate'][()],
            fp['sweepSpec/freqCtr'][()], 
            fp['sweepSpec/scale'][()],
            fp['sweepSpec/apod/duration'][()],
            fp['sweepSpec/sweepName'][()]
          ]
#
    mData = dict(zip(keys, dummy))
#---------- Print info
    keys = [
            'name',
            'cable_refractiveIndexes',
            'dim_Names',
            'dimensionUnits',
            'dimensionSizes',
            'dim_unitScale',
            'dt',
            'dx',
            'gaugeLength',
            'sensitivityUnits',
            'sensitivities',
            'sweepRate',
            'sweepFreqCtr',
            'sweepScale',
            'sweepDuration(μs)',
            'sweepName'
            ]
#--- Find the maximum length of parameter strings for alignment
    max_param_len = max(len(p) for p in keys)
    print(f"\n>> {'Parameter':<{max_param_len}} {'Value'}")
    for param in keys:
        print(f"   {param:<{max_param_len}} {mData.get(param)}")
#--- Returns
    return mData
#
# -------------- End of function   ---------------------