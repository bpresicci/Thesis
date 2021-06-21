from scipy import signal as sig
import numpy as np
from scipy import fft

def feature_extraction(cfg, freq_range, nCh, epoch, check_nperseg):
    """
    Returns the PSD computed in the frequency range given as f_band. The algorithm used to compute the PSD
    is exactly the same as the one ScorEpochs uses, except when check_nperseg = True.
    ScorEpochs uses the default value of nperseg, the feature extraction might use a changed value.

    INPUT
        cfg: dictionary with the following key-value pairs
             freqRange    - list with the frequency range used to compute the power spectrum by ScorEpochs (see scipy.stats.spearmanr()
                            function)
             fs           - integer representing sample frequency
             windowL      - integer representing the window length (in seconds)
             smoothFactor - smoothing factor for the power spectrum (0 by default)
             wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0 by
                            default)
        freq_range: list with the frequency range used to compute the PSD for task of pattern recognition
        nCh: integer, total number of channels
        epoch: 3d list of the data divided in equal length epochs of length windowL (epochs X channels X time samples), provided by ScorEpochs
        check_nperseg: boolean, if True used to check whether the default nperseg parameter is usable to compute correctly the Spearman coefficient,
                       if not usable, nperseg will be changed. If False, the default value will be used without checking.

    OUTPUT:
        psd: 3d list containing the computed PSD, has shape: (epochs per user X number of channels X number of PSD samples)
    """
    epLen = cfg['windowL'] * cfg['fs']  # Computes the number of samples of each epoch (for each channel)
    smoothing_condition = 'smoothFactor' in cfg.keys() and cfg['smoothFactor'] > 1  # True if the smoothing has to be executed, 0 otherwise
    nEp = len(epoch)  # Computes the total number of epochs

    segLen = round(epLen/8) # The default value of nperseg
    if check_nperseg:
      check_freqs = fft.rfftfreq(segLen, 1/cfg['fs'])
      check = 0
      for value, i in zip(check_freqs, range(len(check_freqs))):
        if value >= cfg['freqRange'][0] and value <= cfg['freqRange'][1]:
          check += 1
      if check < 9:
        fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
        segLen = round(1.0/(fix * 1/cfg['fs']))
    """
    The array of frequencies returned by sig.welch() depends on the parameter called nperseg, which is the lenght of each segment. Aiming to obtain a sufficient
    number of computed PSD values to be able to calculate the Spearman coefficients, while keeping an observation window lenght (in seconds, cfg['windowL'])
    limited and a restricted frequency band (i.e. the alpha band, [8, 13]), it is necessary to modify the computing of nperseg.
    The function called by sig.welch() to obtain the sample frequencies is fft.rfftfreq(), from the scipy library, but numpy includes an identical
    function that performs the same calculations and its raw code is available, unlike scipy's one.
    In the source code of numpy.fft.rfftfreq() (https://github.com/numpy/numpy/blob/v1.20.0/numpy/fft/helper.py#L172-L221) can be read:

     Parameters
        ----------
        n : int
           Window length.
       d : scalar, optional
           Sample spacing (inverse of the sampling rate). Defaults to 1.
       Returns
       -------
       f : ndarray
           Array of length ``n//2 + 1`` containing the sample frequencies.

     def rfftfreq(n, d=1.0):
       if not isinstance(n, integer_types):
           raise ValueError("n should be an integer")
       val = 1.0/(n*d)
       N = n//2 + 1
       results = arange(0, N, dtype=int)
       return results * val

     Be cfg['freqRange'] = [low, high], if we want that in this band there were at least 9 sample frequencies to compute the PSD with, the value of "val"
     has to be changed, because it is the distance between a frequency and the following one in the array "f" returned by sig.welch()
     (it can be easily verified that f[i] - f[i - 1] = val). "d" is the reciprocal number of "fs", n = nperseg, so it follows:
     
     val = 1/(n*d) = fs/nperseg
     number of sample frequencies in cfg['freqRange'] = (high - low) / val
     
     And:
     
     (high - low) / val >= 9
     val <= (high - low) / 9
     
     Which leads to:
     
     fix = (cfg['freqRange'][1] - cfg['freqRange'][0]) / 9
     val = fs / nperseg -> nperseg = fs/val = 1/(val * d) with d = 1/fs
     
     So:
     
     segLen = 1.0/(fix * 1/cfg['fs'])
     
     This way we can obtain an f array and a pxx from sig.welch() which lenght in the interval defined by cfg['freqRange'] is at least 9.

    """
    for e in range(nEp): # The algorithm to compute the PSD is exactly the same as the one used by ScorEpochs, except for nperseg = segLen
        for c in range(nCh):
            # compute power spectrum
            f, aux_pxx = sig.welch(epoch[e][c].T, cfg['fs'], window='hamming', nperseg=segLen, detrend=False)  # The nperseg allows the MATLAB pwelch correspondence
            if c == 0 and e == 0:  # The various parameters are obtained in the first interation
                psd, idx_min, idx_max, nFreq = _spectrum_parameters(f, freq_range, aux_pxx, nEp, nCh)
                if smoothing_condition:
                    window_range, initial_f, final_f = _smoothing_parameters(cfg['smoothFactor'], nFreq)
            if smoothing_condition:
                psd[e][c] = _movmean(aux_pxx, cfg['smoothFactor'], initial_f, final_f, nFreq, idx_min, idx_max)
            else:
                psd[e][c] = aux_pxx[idx_min:idx_max + 1]  # pxx takes the only interested spectrum-related sub-array
    return psd

def _movmean(aux_pxx, smoothFactor, initial_f, final_f, nFreq, idx_min, idx_max):   #It is not weighted
    """
    Function used for computing the smoothed power spectrum through moving average filter, where each output sample is
    evaluated on the center of the window at each iteration (or the one furthest to the right of the two in the center,
    in case of a window with an even number of elements), without padding on edges (FOR INTERNAL USE ONLY).
     X = [X(0), X(1), X(2), X(3), X(4)]
     smoothFactor = 3
     Y(0) = (X(0))+X(1))/2
     Y(1) = (X(0)+X(1)+X(2))/3
     Y(2) = (X(1)+X(2)+X(3))/3
     Y(3) = (X(2)+X(3)+X(4))/3
     Y(4) = (X(3)+X(4))/2
    """
    smoothed = np.zeros((idx_max-idx_min+1,))
    for f in range(nFreq):
        if f < initial_f:
            smoothed[f] = np.mean(aux_pxx[idx_min:idx_min+f+initial_f+1])
        elif f >= final_f:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:])
        elif smoothFactor % 2 == 0:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f])
        else:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f+1])
    return smoothed


def _spectrum_parameters(f, freqRange, aux_pxx, nEp, nCh):
    """
    Function which defines the spectrum parameters for the scorEpochs function (FOR INTERNAL USE ONLY).
    """
    idx_min = int(np.argmin(abs(f - freqRange[0])))
    idx_max = int(np.argmin(abs(f - freqRange[-1])))
    nFreq = len(aux_pxx[idx_min:idx_max + 1])
    pxx = np.zeros((nEp, nCh, nFreq))
    return pxx, idx_min, idx_max, nFreq


def _smoothing_parameters(smoothFactor, nFreq):
    """
    Function which defines the parameters of the window to be used by the scorEpochs function in smoothing the spectrum
    (FOR INTERNAL USE ONLY).
    """
    window_range = round(smoothFactor)
    initial_f = int(window_range/2)
    final_f = nFreq - initial_f
    return window_range, initial_f, final_f

