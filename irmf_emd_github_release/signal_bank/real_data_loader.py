
import numpy as np

def load_real_signal_csv(path,fs=None):
    data=np.loadtxt(path,delimiter=',')
    if data.ndim==1:
        if fs is None: raise ValueError('fs required for one-column CSV')
        return np.arange(len(data))/fs, data.astype(float), fs
    T=data[:,0].astype(float); Y=data[:,1].astype(float)
    if fs is None: fs=1/np.median(np.diff(T))
    return T,Y,fs
