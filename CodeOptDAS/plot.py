#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________System Utilities__________/
import obspy
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
"""
Pltot f-k (frequency-wavenumber) transform of DAS strain-rate 
    Plot the f-k amplitude spectrum in dB (relative to its maximum).

    velocities : optional list of apparent velocities [m/s]; draws the lines f = c*k
"""
def plot_fk(fk, f, k, fmax=None, kmax=None, dyn_range_db=60, velocities=None,
            cmap="turbo", ax=None, title = None):
    amp = fk.abs()
    amp_db = 20 * torch.log10(amp / amp.max() + 1e-12)
    amp_db = amp_db.cpu().numpy()
    f_np, k_np = f.cpu().numpy(), k.cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        amp_db, origin="lower", aspect="auto", cmap=cmap,
        extent=[k_np[0], k_np[-1], f_np[0], f_np[-1]],
        vmin=-dyn_range_db, vmax=0,
    )
    if velocities is not None:
        kk = np.array([k_np[0], k_np[-1]])
        for c in velocities:
            ax.plot(kk, c * kk, "w--", lw=0.8)
            ax.text(0.02 * k_np[-1], c * 0.02 * k_np[-1], f"{c:g} m/s",
                    color="w", fontsize=8, va="bottom")

    ax.set_xlim(-kmax if kmax else k_np[0], kmax if kmax else k_np[-1])
    ax.set_ylim(0, fmax if fmax else f_np[-1])
    ax.set_xlabel("Wavenumber k (cycles/m)")
    ax.set_ylabel("Frequency f (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Amplitude (dB)")
    plt.tight_layout()
#    plt.savefig("fk_spectrum.png", dpi=150)
#    return ax
    plt.show(block=True)
# -------------- End of function   ----------------------------
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
    strain_rate : ndarray [nt, nx]
        Strain-rate data.
    x : ndarray [nx]
        Distance along fiber (m).
    z : ndarray [nx]
        Bathymetry (m).
    t : ndarray [nt]
        Time vector.
    mData : dict
        Metadata dictionary containing:
            gaugeLength
            dx
    title : str
        Figure title.
"""

def plotTS(strain_rate, x, z, t, mData, title='Strain rate'):
     # DAS decimation according to gauge length
     step = max(1, int(mData['gaugeLength'] / mData['dx']))

     x_plot = x[::step]
     z_plot = z[::step]
     data_plot = strain_rate[:, ::step]

     # ------------------------------------------------------------------
     # Create figure with two panels
     # ------------------------------------------------------------------
     fig, (ax_bathy, ax_das) = plt.subplots(
        2, 1,
        figsize=(20, 10),
        sharex=True,
        gridspec_kw={
            'height_ratios': [1, 7],    #1, 5
            'hspace': 0.05
        }
     )

     # ------------------------------------------------------------------
     # Bathymetry
     # ------------------------------------------------------------------
     ax_bathy.plot(x_plot, z_plot, 'k', lw=1.5)

     """ax_bathy.fill_between(
        x_plot,
        z_plot,
        np.nanmin(z_plot),
        color='lightgray',
        alpha=0.7
     )
     """
     ax_bathy.set_ylabel('Depth (m)')

     # invert bathymetry axis
#     ax_bathy.invert_yaxis()

     ax_bathy.grid(alpha=0.3)

     # cleaner appearance
     ax_bathy.tick_params(axis='x', labelbottom=False)
     ax_bathy.spines['right'].set_visible(True)
     ax_bathy.spines['top'].set_visible(True)

     # ------------------------------------------------------------------
     # DAS image
     # ------------------------------------------------------------------
     clipVal = np.percentile(np.abs(data_plot), 99)

     im = ax_das.imshow(
        data_plot.T,
        extent=[x_plot[0], x_plot[-1], t[-1], t[0]],
        aspect='auto',
        cmap='seismic',
        vmin=-clipVal,
        vmax=clipVal
     )

     ax_das.set_xlabel('Distance (m)')
     ax_das.set_ylabel('Time (s)')
#     ax_das.set_title(title)

     ax_das.grid(alpha=0.2)

     # ------------------------------------------------------------------
     # Colorbar
     # ------------------------------------------------------------------
     divider = make_axes_locatable(ax_das)
     cax = divider.append_axes("right", size="2%", pad=0.1)

     cbar = plt.colorbar(im, cax=cax)
     cbar.set_label(title)

     # after plotting everything
     ax_bathy.set_xlim(ax_das.get_xlim())

     """cbar = fig.colorbar(
         im,
         ax=[ax_bathy, ax_das],
         pad=0.02,
         aspect=30
     )
     """

     fig.tight_layout()

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
def plotTxCor(strain_rate, cc, max_lag_npts, y, dt, srcD, z, title=None):
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
     
     # Add a second vertical axis on the right displaying depth 'z' alongside distance 'y'
     ax2 = ax.twinx()
     ax2.set_ylim(y[0], y[-1])
     
     # Map the y ticks to their corresponding z depth values
     y_ticks = ax.get_yticks()
     # Filter ticks to keep them within the actual y range
     y_ticks = y_ticks[(y_ticks >= y[0]) & (y_ticks <= y[-1])]
     ax2.set_yticks(y_ticks)
     
     # Interpolate z values at the tick positions
     z_ticks = np.interp(y_ticks, y, z)
     ax2.set_yticklabels([f"{val:.1f}" for val in z_ticks])
     ax2.set_ylabel('Depth (m)')
     
     plt.tight_layout()
#     plt.show()
     plt.show(block=True)

# -------------- plotTxCor ----------------------------
"""
Plot linear bathimetry along FO
"""
def pltFOdepth(s, z):
    plt.figure(figsize=(8,6))
    # Cubic fit
    plt.plot(s/1000., z, 'r', linewidth=2)

    plt.xlabel('Distance along FO (km)')
    plt.ylabel('Depth (m)')
    plt.title('Linear bathimetry (R2=91.8%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)