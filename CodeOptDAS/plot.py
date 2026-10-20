#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________System Utilities__________/
import obspy
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


#+────────────────────────────────── pltpGram ─────────────────────────────────────────────+
"""Plot periodogram

Parameters:
-----------
data :         numpy.ndarray
       2-D array of strain data with shape (nt, nx), where axis 0 is time and axis 1 is spatial channels.
periodogram : numpy.ndarray w/ shape: [n_channels, n_freqs, n_time_steps]  Averaged periodogram
pltPar:      List of ploting paramters
       
Returns:
--------

"""
def pltpGram(t, data, periodogram, pltPar, n_fft=None, hop_length= None):
#--- Total time window ΔT
     deltaT = t.size * (t[1] - t[0])
#--- Unpack pltPar
     time_steps, t_stft, freqs , channel_idx, xGl, f0, f1 = pltPar
#--- Determine appropriate time unit for the horizontal axis
     if deltaT < 60:
         t_raw = t
         t_spectrogram = t_stft
         time_label = "Time (s)"
         t_max = deltaT
     elif deltaT < 3600:
         t_raw = t / 60.0
         t_spectrogram = t_stft / 60.0
         time_label = "Time (min)"
         t_max = deltaT / 60.0
     else:
         t_raw = t / 3600.0
         t_spectrogram = t_stft / 3600.0
         time_label = "Time (h)"
         t_max = deltaT / 3600.0
#---------- Plot the Periodogram
     fig = plt.figure(figsize=(11, 8))
#--- Set up a grid layout to separate the colorbar and keep main plots aligned
     gs = gridspec.GridSpec(2, 2, width_ratios=[40, 1], height_ratios=[1, 2], hspace=0.15)
#--- Main plotting axes sharing the x-axis
     ax1 = fig.add_subplot(gs[0, 0])
     ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
#--- Colorbar axes
     cax = fig.add_subplot(gs[1, 1])
#--- Top panel: Strain Rate Time-Series at channel 0.
     ax1.plot(t_raw, data[:, channel_idx], color='royalblue', lw=0.7)    #, label=f'Channel {channel_idx}'
     ax1.set_title(f"∂ε/∂t (Channel {channel_idx} at {xGl:.1f}km)", fontsize=14)
     ax1.set_ylabel("∂ε/∂t", fontsize=11)
     ax1.grid(True, alpha=0.3)
#     ax1.legend(loc='upper right')
#--- Bottom panel: STFT averaged periodogram across all channels
     im = ax2.pcolormesh(
         t_spectrogram,
         freqs,
         10 * np.log10(periodogram + 1e-10),
         shading='gouraud',
         cmap='viridis'
     )
     ax2.set_ylabel("Frequency (Hz)", fontsize=11)
     ax2.set_xlabel(time_label, fontsize=11)
     ax2.grid(True, alpha=0.3)
#--- Freq limits
     ax2.set_ylim(f0, f1)
#--- Hide x-axis tick labels for the top panel to avoid redundancy
     plt.setp(ax1.get_xticklabels(), visible=False)
#--- Force both x-axes to have the exact same limits
     ax1.set_xlim(0, t_max)
#--- Plot colorbar in its dedicated axis so it doesn't affect main plot dimensions
     cbar = fig.colorbar(im, cax=cax, orientation='vertical')
     cbar.set_label("PSD (dB)", fontsize=10)

     plt.show(block=True)
# -------------- End of function   ----------------------------
# -------------- pltFO  ----------------------------
""" Function to plot the FO
    data = lat, lon, depth
"""
#
def pltFO(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = data[:, 0] / 1000.
    y_coords = data[:, 1] / 1000.
    z_coords = data[:, 2]

    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', label='Points in A')
    ax.plot(x_coords, y_coords, z_coords, color='red', label='Path') # Plotting the path

    ax.set_xlabel('X (N→S) km')
    ax.set_ylabel('Y (E→W) km')
    ax.set_zlabel('Z (Depth) m')
    ax.invert_zaxis()
    ax.set_title('OptoDAS')
    ax.legend()
#    plt.show()
    plt.show(block=True)
#
# -------------- End of function   ---------------------
# -------------- plotTS ----------------------------
"""
A time series plotting script
    Parameters
    ----------
    strRate: Numpy matrix(n_times, n_cha)

    t:       Time vector
    mData.get('dt'): temporal sampling
    Returns
    -------
    :
"""
def plotTS(strain_rate, x, t, mData, title = 'Strain rate'):
#
#--- Local function to plot the DAS data: data = [nt, nx]
     def show_data(data, das_time, pclip=99.5, label='∂ε/∂t'):
         fig, ax1 = plt.subplots(figsize=(20,10))
         clipVal = np.percentile(np.absolute(data), pclip)
         im = ax1.imshow(data.T, 
                     extent=[x[0], x[-1], t[-1], t[0]],          #0, data.shape[0]
                     aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))
#
         ax1.set_xlabel("Distance (m)")
         ax1.set_ylabel("Time [s]")

         ax1.grid()
         divider = make_axes_locatable(ax1)
         cax = divider.append_axes("right", size="2%", pad=0.7)
         cbar = plt.colorbar(im, orientation="vertical", cax=cax)
         cbar.set_label(label)
         return fig, ax1
#
#--- Strain rate
     step = int(mData['gaugeLength'] / mData['dx'])
     fig, ax2 = show_data(strain_rate[:, ::step], t, pclip=99, label="nStrain rate [1/s]")
     ax2.set_title('Strain rate')
     ax2.set_ylim(t[-1], t[0])
     ax2.set_title(title)
#     ax2.plot(ch_list,P_phase_time_taup,"r--",lw=4,label="P-wave")
#     ax2.plot(ch_list,S_phase_time_taup,"c--",lw=4,label="S-wave")
#     ax2.legend()
#
#--- Time series
     """fig, ax = plt.subplots(figsize=(20, 12))
     gain=1.        # 1.5e-1
     for ii in range(len(x)):
         ax.plot(t, strain_rate[:, ::step*10]*gain + ii, "k-", linewidth=2.0)
#
     ax.set_ylim(x[0], x[-1])
     ax.set_xlim(t[0], t[-1])
     ax.set_xlabel('Time (s)')
     ax.set_ylabel('Distance (m)')
     ax.set_title('Strain rate')
     ax.grid()
     """
     plt.show(block=True)
# -------------- plotTxCor ----------------------------
"""
A X-cor plotting script
    Parameters
    ----------
    strRate: Numpy matrix(n_cha, n_times)

    t:       Time vector
    mData.get('dt'): temporal sampling
    Returns
    -------
    :
"""
def plotTxCor(strain_rate, cc, max_lag_npts, y, dt, srcD, title=None):
     fig, ax = plt.subplots(figsize=(15, 10))
     t = np.arange(-max_lag_npts, max_lag_npts+1) * dt
     nch = strain_rate.shape[0]
#     y = np.arange(0, nch)
     vlim = np.percentile(np.abs(cc), 99)
     ratio = (t[-1]-t[0])/(y[-1]-y[0])
     im = ax.imshow(cc, cmap=plt.get_cmap("RdBu"), 
                    extent=[t[0], t[-1], y[0], y[-1]], vmin=-vlim, 
                    vmax=vlim, origin="lower", aspect=ratio)
#
     cbar = plt.colorbar(im, ax=ax)
     cbar.set_label("X-correlation")
     ax.grid()
#
     ax.set_xlabel('Time lag (s)')
     ax.set_ylabel('Distance (m)')
     ax.set_title(title)
     plt.tight_layout()
#     plt.show()
     plt.show(block=True)