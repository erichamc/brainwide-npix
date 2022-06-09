"""
This is mostly functionality from IBL's IO sglx module.
Taken from: https://int-brain-lab.github.io/iblenv/_modules/ibllib/io/spikeglx.htm
and adapted by Ethan Richman to remove some unneeded functionality.


Neuropixel geometry functions adapted from:
https://int-brain-lab.github.io/iblenv/_modules/ibllib/ephys/neuropixel.html
"""

import json
from pathlib import Path
import re
import shutil

import numpy as np

SAMPLE_SIZE = 2  # int16
DEFAULT_BATCH_SIZE = 1e6
# provided as convenience if no meta-data is provided, always better to read from meta
S2V_AP = 2.34375e-06
S2V_LFP = 4.6875e-06

TIP_SIZE_UM = 200

SYNC_PIN_OUT = {'3A': {"pin01": 0,
                       "pin02": 1,
                       "pin03": 2,
                       "pin04": 3,
                       "pin05": None,
                       "pin06": 4,
                       "pin07": 5,
                       "pin08": 6,
                       "pin09": 7,
                       "pin10": None,
                       "pin11": 8,
                       "pin12": 9,
                       "pin13": 10,
                       "pin14": 11,
                       "pin15": None,
                       "pin16": 12,
                       "pin17": 13,
                       "pin18": 14,
                       "pin19": 15,
                       "pin20": None,
                       "pin21": None,
                       "pin22": None,
                       "pin23": None,
                       "pin24": None
                       },
                '3B': {"P0.0": 0,
                       "P0.1": 1,
                       "P0.2": 2,
                       "P0.3": 3,
                       "P0.4": 4,
                       "P0.5": 5,
                       "P0.6": 6,
                       "P0.7": 7,
                       }
                }

# after moving to ks2.5, this should be deprecated
SITES_COORDINATES = np.array([
    [43., 20.],
    [11., 20.],
    [59., 40.],
    [27., 40.],
    [43., 60.],
    [11., 60.],
    [59., 80.],
    [27., 80.],
    [43., 100.],
    [11., 100.],
    [59., 120.],
    [27., 120.],
    [43., 140.],
    [11., 140.],
    [59., 160.],
    [27., 160.],
    [43., 180.],
    [11., 180.],
    [59., 200.],
    [27., 200.],
    [43., 220.],
    [11., 220.],
    [59., 240.],
    [27., 240.],
    [43., 260.],
    [11., 260.],
    [59., 280.],
    [27., 280.],
    [43., 300.],
    [11., 300.],
    [59., 320.],
    [27., 320.],
    [43., 340.],
    [11., 340.],
    [59., 360.],
    [27., 360.],
    [11., 380.],
    [59., 400.],
    [27., 400.],
    [43., 420.],
    [11., 420.],
    [59., 440.],
    [27., 440.],
    [43., 460.],
    [11., 460.],
    [59., 480.],
    [27., 480.],
    [43., 500.],
    [11., 500.],
    [59., 520.],
    [27., 520.],
    [43., 540.],
    [11., 540.],
    [59., 560.],
    [27., 560.],
    [43., 580.],
    [11., 580.],
    [59., 600.],
    [27., 600.],
    [43., 620.],
    [11., 620.],
    [59., 640.],
    [27., 640.],
    [43., 660.],
    [11., 660.],
    [59., 680.],
    [27., 680.],
    [43., 700.],
    [11., 700.],
    [59., 720.],
    [27., 720.],
    [43., 740.],
    [11., 740.],
    [59., 760.],
    [43., 780.],
    [11., 780.],
    [59., 800.],
    [27., 800.],
    [43., 820.],
    [11., 820.],
    [59., 840.],
    [27., 840.],
    [43., 860.],
    [11., 860.],
    [59., 880.],
    [27., 880.],
    [43., 900.],
    [11., 900.],
    [59., 920.],
    [27., 920.],
    [43., 940.],
    [11., 940.],
    [59., 960.],
    [27., 960.],
    [43., 980.],
    [11., 980.],
    [59., 1000.],
    [27., 1000.],
    [43., 1020.],
    [11., 1020.],
    [59., 1040.],
    [27., 1040.],
    [43., 1060.],
    [11., 1060.],
    [59., 1080.],
    [27., 1080.],
    [43., 1100.],
    [11., 1100.],
    [59., 1120.],
    [27., 1120.],
    [11., 1140.],
    [59., 1160.],
    [27., 1160.],
    [43., 1180.],
    [11., 1180.],
    [59., 1200.],
    [27., 1200.],
    [43., 1220.],
    [11., 1220.],
    [59., 1240.],
    [27., 1240.],
    [43., 1260.],
    [11., 1260.],
    [59., 1280.],
    [27., 1280.],
    [43., 1300.],
    [11., 1300.],
    [59., 1320.],
    [27., 1320.],
    [43., 1340.],
    [11., 1340.],
    [59., 1360.],
    [27., 1360.],
    [43., 1380.],
    [11., 1380.],
    [59., 1400.],
    [27., 1400.],
    [43., 1420.],
    [11., 1420.],
    [59., 1440.],
    [27., 1440.],
    [43., 1460.],
    [11., 1460.],
    [59., 1480.],
    [27., 1480.],
    [43., 1500.],
    [11., 1500.],
    [59., 1520.],
    [43., 1540.],
    [11., 1540.],
    [59., 1560.],
    [27., 1560.],
    [43., 1580.],
    [11., 1580.],
    [59., 1600.],
    [27., 1600.],
    [43., 1620.],
    [11., 1620.],
    [59., 1640.],
    [27., 1640.],
    [43., 1660.],
    [11., 1660.],
    [59., 1680.],
    [27., 1680.],
    [43., 1700.],
    [11., 1700.],
    [59., 1720.],
    [27., 1720.],
    [43., 1740.],
    [11., 1740.],
    [59., 1760.],
    [27., 1760.],
    [43., 1780.],
    [11., 1780.],
    [59., 1800.],
    [27., 1800.],
    [43., 1820.],
    [11., 1820.],
    [59., 1840.],
    [27., 1840.],
    [43., 1860.],
    [11., 1860.],
    [59., 1880.],
    [27., 1880.],
    [11., 1900.],
    [59., 1920.],
    [27., 1920.],
    [43., 1940.],
    [11., 1940.],
    [59., 1960.],
    [27., 1960.],
    [43., 1980.],
    [11., 1980.],
    [59., 2000.],
    [27., 2000.],
    [43., 2020.],
    [11., 2020.],
    [59., 2040.],
    [27., 2040.],
    [43., 2060.],
    [11., 2060.],
    [59., 2080.],
    [27., 2080.],
    [43., 2100.],
    [11., 2100.],
    [59., 2120.],
    [27., 2120.],
    [43., 2140.],
    [11., 2140.],
    [59., 2160.],
    [27., 2160.],
    [43., 2180.],
    [11., 2180.],
    [59., 2200.],
    [27., 2200.],
    [43., 2220.],
    [11., 2220.],
    [59., 2240.],
    [27., 2240.],
    [43., 2260.],
    [11., 2260.],
    [59., 2280.],
    [43., 2300.],
    [11., 2300.],
    [59., 2320.],
    [27., 2320.],
    [43., 2340.],
    [11., 2340.],
    [59., 2360.],
    [27., 2360.],
    [43., 2380.],
    [11., 2380.],
    [59., 2400.],
    [27., 2400.],
    [43., 2420.],
    [11., 2420.],
    [59., 2440.],
    [27., 2440.],
    [43., 2460.],
    [11., 2460.],
    [59., 2480.],
    [27., 2480.],
    [43., 2500.],
    [11., 2500.],
    [59., 2520.],
    [27., 2520.],
    [43., 2540.],
    [11., 2540.],
    [59., 2560.],
    [27., 2560.],
    [43., 2580.],
    [11., 2580.],
    [59., 2600.],
    [27., 2600.],
    [43., 2620.],
    [11., 2620.],
    [59., 2640.],
    [27., 2640.],
    [11., 2660.],
    [59., 2680.],
    [27., 2680.],
    [43., 2700.],
    [11., 2700.],
    [59., 2720.],
    [27., 2720.],
    [43., 2740.],
    [11., 2740.],
    [59., 2760.],
    [27., 2760.],
    [43., 2780.],
    [11., 2780.],
    [59., 2800.],
    [27., 2800.],
    [43., 2820.],
    [11., 2820.],
    [59., 2840.],
    [27., 2840.],
    [43., 2860.],
    [11., 2860.],
    [59., 2880.],
    [27., 2880.],
    [43., 2900.],
    [11., 2900.],
    [59., 2920.],
    [27., 2920.],
    [43., 2940.],
    [11., 2940.],
    [59., 2960.],
    [27., 2960.],
    [43., 2980.],
    [11., 2980.],
    [59., 3000.],
    [27., 3000.],
    [43., 3020.],
    [11., 3020.],
    [59., 3040.],
    [43., 3060.],
    [11., 3060.],
    [59., 3080.],
    [27., 3080.],
    [43., 3100.],
    [11., 3100.],
    [59., 3120.],
    [27., 3120.],
    [43., 3140.],
    [11., 3140.],
    [59., 3160.],
    [27., 3160.],
    [43., 3180.],
    [11., 3180.],
    [59., 3200.],
    [27., 3200.],
    [43., 3220.],
    [11., 3220.],
    [59., 3240.],
    [27., 3240.],
    [43., 3260.],
    [11., 3260.],
    [59., 3280.],
    [27., 3280.],
    [43., 3300.],
    [11., 3300.],
    [59., 3320.],
    [27., 3320.],
    [43., 3340.],
    [11., 3340.],
    [59., 3360.],
    [27., 3360.],
    [43., 3380.],
    [11., 3380.],
    [59., 3400.],
    [27., 3400.],
    [11., 3420.],
    [59., 3440.],
    [27., 3440.],
    [43., 3460.],
    [11., 3460.],
    [59., 3480.],
    [27., 3480.],
    [43., 3500.],
    [11., 3500.],
    [59., 3520.],
    [27., 3520.],
    [43., 3540.],
    [11., 3540.],
    [59., 3560.],
    [27., 3560.],
    [43., 3580.],
    [11., 3580.],
    [59., 3600.],
    [27., 3600.],
    [43., 3620.],
    [11., 3620.],
    [59., 3640.],
    [27., 3640.],
    [43., 3660.],
    [11., 3660.],
    [59., 3680.],
    [27., 3680.],
    [43., 3700.],
    [11., 3700.],
    [59., 3720.],
    [27., 3720.],
    [43., 3740.],
    [11., 3740.],
    [59., 3760.],
    [27., 3760.],
    [43., 3780.],
    [11., 3780.],
    [59., 3800.],
    [43., 3820.],
    [11., 3820.],
    [59., 3840.],
    [27., 3840.]])

NC = 384

class Neuropixel():
    def __init__(self):
        return
    
    def rc2xy(self, row, col, version=1):
        """
        converts the row/col indices to um coordinates.
        :param row: row index on the probe
        :param col: col index on the probe
        :param version: neuropixel major version 1 or 2
        :return: dictionary with keys x and y
        """
        if version == 1:
            x = col * 16 + 11
            y = (row * 20) + 20
        elif np.floor(version) == 2:
            x = col * 32
            y = row * 15
        return {'x': x, 'y': y}


    def dense_layout(self, version=1):
        """
        Returns a dense layout indices map for neuropixel, as used at IBL
        :param version: major version number: 1 or 2 or 2.4
        :return: dictionary with keys 'ind', 'col', 'row', 'x', 'y'
        """
        ch = {'ind': np.arange(NC),
            'row': np.floor(np.arange(NC) / 2),
            'shank': np.zeros(NC)}

        if version == 2:
            ch.update({'col': np.tile(np.array([0, 1]), int(NC / 2))})
        elif version == 2.4:
            # the 4 shank version default is rather complicated
            shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
            shank_row = np.tile(shank_row, 8)
            shank_row += np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))).flatten() * 24
            ch.update({
                'col': np.tile(np.array([0, 1]), int(NC / 2)),
                'shank': np.tile(np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))).flatten(),
                'row': shank_row})
        elif version == 1:
            ch.update({'col': np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
        # for all, get coordinates
        ch.update(self.rc2xy(ch['row'], ch['col'], version=version))
        return ch



    def adc_shifts(self, version=1):
        """
        The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
        The ADC to channel mapping is done per odd and even channels:
        ADC1: ch1, ch3, ch5, ch7...
        ADC2: ch2, ch4, ch6....
        ADC3: ch33, ch35, ch37...
        ADC4: ch34, ch36, ch38...
        Therefore, channels 1, 2, 33, 34 get sample at the same time. I hope this is more or
        less clear. In 1.0, it is similar, but there we have 32 ADC that sample each 12 channels."
        - Nick on Slack after talking to Carolina - ;-)
        """
        if version == 1:
            adc_channels = 12
            # version 1 uses 32 ADC that sample 12 channels each
        elif np.floor(version) == 2:
            # version 2 uses 24 ADC that sample 16 channels each
            adc_channels = 16
        adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
        sample_shift = np.zeros_like(adc)
        for a in adc:
            sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
        return sample_shift, adc



    def trace_header(self, version=1):
        """
        Returns the channel map for the dense layout used at IBL
        :param version: major version number: 1 or 2
        :return: , returns a dictionary with keys
        x, y, row, col, ind, adc and sampleshift vectors corresponding to each site
        """
        h = self.dense_layout(version=version)
        h['sample_shift'], h['adc'] = self.adc_shifts(version=version)
        return h


neuropixel = Neuropixel()


class Reader:
    """
    Class for SpikeGLX reading purposes
    Some format description was found looking at the Matlab SDK here
    https://github.com/billkarsh/SpikeGLX/blob/master/MATLAB-SDK/DemoReadSGLXData.m

    To open a spikeglx file that has an associated meta-data file:
    sr = spikeglx.Reader(bin_file_path)

    To open a flat binary file:

    sr = spikeglx.Reader(bin_file_path, nc=385, ns=nsamples, fs=30000)
    one can provide more options to the reader:
    sr = spikeglx.Reader(..., dtype='int16, s2mv=2.34375e-06)

    usual sample 2 mv conversion factors:
        s2mv = 2.34375e-06 (NP1 ap banc) : default value used
        s2mv = 4.6875e-06 (NP1 lfp band)

    Note: To release system resources the close method must be called
    """

    def __init__(self, sglx_file, open=True, nc=None, ns=None, fs=None, dtype='int16', s2v=None,
                 nsync=None):
        """
        An interface for reading data from a SpikeGLX file
        :param sglx_file: Path to a SpikeGLX file (compressed or otherwise)
        :param open: when True the file is opened
        """
        self.file_bin = Path(sglx_file)
        self.nbytes = self.file_bin.stat().st_size
        self.dtype = np.dtype(dtype)
        file_meta_data = Path(sglx_file).with_suffix('.meta')
        if not file_meta_data.exists():
            err_str = "Instantiating an Reader without meta data requires providing nc, fs and nc parameters"
            assert (nc is not None and fs is not None and nc is not None), err_str
            self.file_meta_data = None
            self.meta = None
            self._nc, self._fs, self._ns = (nc, fs, ns)
            # handles default parameters: if int16 we assume it's a raw recording with 1 sync and sample2mv
            # if its' float32 or something else, we assume the sync channel has been removed and the scaling applied
            if nsync is None:
                nsync = 1 if self.dtype == np.dtype('int16') else 0
            self._nsync = nsync
            if s2v is None:
                s2v = S2V_AP if self.dtype == np.dtype('int16') else 1.0
            self.channel_conversion_sample2v = {'samples': np.ones(nc) * s2v}
            self.channel_conversion_sample2v['samples'][-nsync:] = 1
        else:
            # normal case we continue reading and interpreting the metadata file
            self.file_meta_data = file_meta_data
            self.meta = read_meta_data(file_meta_data)
            self.channel_conversion_sample2v = _conversion_sample2v_from_meta(self.meta)
            self._raw = None
        if open:
            self.open()

    def open(self):
        sglx_file = str(self.file_bin)
        
        if self.nc * self.ns * self.dtype.itemsize != self.nbytes:
            ftsec = self.file_bin.stat().st_size / self.dtype.itemsize / self.nc / self.fs
            _logger.warning(f"{sglx_file} : meta data and filesize do not checkout\n"
                            f"File size: expected {self.meta['fileSizeBytes']},"
                            f" actual {self.file_bin.stat().st_size}\n"
                            f"File duration: expected {self.meta['fileTimeSecs']},"
                            f" actual {ftsec}\n"
                            f"Will attempt to fudge the meta-data information.")
            self.meta['fileTimeSecs'] = ftsec
        self._raw = np.memmap(sglx_file, dtype=self.dtype, mode='r', shape=(self.ns, self.nc))


    def close(self):
        if self.is_open:
            getattr(self._raw, '_mmap', self._raw).close()


    def __enter__(self):
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.read(nsel=item, sync=False)
        elif len(item) == 2:
            return self.read(nsel=item[0], csel=item[1], sync=False)

    @property
    def sample2volts(self):
        return self.channel_conversion_sample2v[self.type]

    @property
    def geometry(self):
        """
        Gets the geometry, ie. the full trace header for the recording
        :return: dictionary with keys 'row', 'col', 'ind', 'shank', 'adc', 'x', 'y', 'sample_shift'
        """
        return _geometry_from_meta(self.meta)

    @property
    def shape(self):
        return self.ns, self.nc

    @property
    def is_open(self):
        return self._raw is not None

    @property
    def version(self):
        """Gets the version string: '3A', '3B2', '3B1', 'NP2.1', 'NP2.4'"""
        return None if self.meta is None else _get_neuropixel_version_from_meta(self.meta)

    @property
    def major_version(self):
        """Gets the the major version int: 1 or 2"""
        return None if self.meta is None else _get_neuropixel_major_version_from_meta(self.meta)

    @property
    def rl(self):
        return self.ns / self.fs

    @property
    def type(self):
        """:return: ap, lf or nidq. Useful to index dictionaries """
        if not self.meta:
            return 'samples'
        return _get_type_from_meta(self.meta)

    @property
    def fs(self):
        """ :return: sampling frequency (Hz) """
        return self._fs if self.meta is None else _get_fs_from_meta(self.meta)

    @property
    def nc(self):
        """ :return: number of channels """
        return self._nc if self.meta is None else _get_nchannels_from_meta(self.meta)

    @property
    def nsync(self):
        """:return: number of sync channels"""
        return self._nsync if self.meta is None else len(_get_sync_trace_indices_from_meta(self.meta))

    @property
    def ns(self):
        """ :return: number of samples """
        if self.meta is None:
            return self._ns
        return int(np.round(self.meta.get('fileTimeSecs') * self.fs))


    def read(self, nsel=slice(0, 10000), csel=slice(None), sync=True):
        """
        Read from slices or indexes
        :param slice_n: slice or sample indices
        :param slice_c: slice or channel indices
        :return: float32 array
        """
        if not self.is_open:
            raise IOError('Reader not open; call `open` before `read`')
        darray = self._raw[nsel, csel].astype(np.float32, copy=True)
        darray *= self.channel_conversion_sample2v[self.type][csel]
        if sync:
            return darray, self.read_sync(nsel)
        else:
            return darray


    def read_samples(self, first_sample=0, last_sample=10000, channels=None):
        """
        reads all channels from first_sample to last_sample, following numpy slicing convention
        sglx.read_samples(first=0, last=100) would be equivalent to slicing the array D
        D[:,0:100] where the last axis represent time and the first channels.

         :param first_sample: first sample to be read, python slice-wise
         :param last_sample:  last sample to be read, python slice-wise
         :param channels: slice or numpy array of indices
         :return: numpy array of int16
        """
        if channels is None:
            channels = slice(None)
        return self.read(slice(first_sample, last_sample), channels)


    def read_sync_digital(self, _slice=slice(0, 10000)):
        """
        Reads only the digital sync trace at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_digital(slice(0,10000))
        """
        if not self.is_open:
            raise IOError('Reader not open; call `open` before `read`')
        if not self.meta:
            _logger.warning('Sync trace not labeled in metadata. Assuming last trace')
        return split_sync(self._raw[_slice, _get_sync_trace_indices_from_meta(self.meta)])


    def read_sync_analog(self, _slice=slice(0, 10000)):
        """
        Reads only the analog sync traces at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_analog(slice(0,10000))
        """
        if not self.meta:
            return
        csel = _get_analog_sync_trace_indices_from_meta(self.meta)
        if not csel:
            return
        else:
            return self.read(nsel=_slice, csel=csel, sync=False)


    def read_sync(self, _slice=slice(0, 10000), threshold=1.2, floor_percentile=10):
        """
        Reads all sync trace. Convert analog to digital with selected threshold and append to array
        :param _slice: samples slice
        :param threshold: (V) threshold for front detection, defaults to 1.2 V
        :param floor_percentile: 10% removes the percentile value of the analog trace before
         thresholding. This is to avoid DC offset drift
        :return: int8 array
        """
        digital = self.read_sync_digital(_slice)
        analog = self.read_sync_analog(_slice)
        if analog is not None and floor_percentile:
            analog -= np.percentile(analog, 10, axis=0)
        if analog is None:
            return digital
        analog[np.where(analog < threshold)] = 0
        analog[np.where(analog >= threshold)] = 1
        return np.concatenate((digital, np.int8(analog)), axis=1)



def read(sglx_file, first_sample=0, last_sample=10000):
    """
    Function to read from a spikeglx binary file without instantiating the class.
    Gets the meta-data as well.

    >>> ibllib.io.spikeglx.read('/path/to/file.bin', first_sample=0, last_sample=1000)

    :param sglx_file: full path the the binary file to read
    :param first_sample: first sample to be read, python slice-wise
    :param last_sample: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with Reader(sglx_file) as sglxr:
        D, sync = sglxr.read_samples(first_sample=first_sample, last_sample=last_sample)
    return D, sync, sglxr.meta


def read_meta_data(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values

    :param md_file: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with open(md_file) as fid:
        md = fid.read()
        d = {}
        for a in md.splitlines():
            parts = a.split('=')
            try:
                k,v = parts
                # if all numbers, try to interpret the string
                if v and re.fullmatch('[0-9,.]*', v) and v.count('.') < 2:
                    v = [float(val) for val in v.split(',')]
                    # scalars should not be nested
                    if len(v) == 1:
                        v = v[0]
                # tildes in keynames removed
                d[k.replace('~', '')] = v
            except:
                pass
    d['neuropixelVersion'] = _get_neuropixel_version_from_meta(d)
    d['serial'] = _get_serial_number_from_meta(d)
    return d


def write_meta_data(md, md_file):
    """
    Parses a dict into a spikeglx meta data file
    TODO write a test for this function, (read in, write out and make sure it is the same)
    :param meta: meta data dict
    :param md_file: file to save meta data to
    :return:
    """
    with open(md_file, 'w') as fid:
        for key, val in md.items():
            if isinstance(val, list):
                val = ','.join([str(int(v)) for v in val])
            if isinstance(val, float):
                if val.is_integer():
                    val = int(val)
            fid.write(f'{key}={val}\n')


def _get_savedChans_subset(chns):
    """
    Get the subset of the original channels that are saved per shank
    :param chns:
    :return:
    """
    chn_grps = np.r_[0, np.where(np.diff(chns) != 1)[0] + 1, len(chns)]
    chn_subset = [f'{chns[chn_grps[i]]}:{chns[chn_grps[i + 1] - 1]}'
                  if chn_grps[i] < len(chns) - 1 else f'{chns[chn_grps[i]]}'
                  for i in range(len(chn_grps) - 1)]

    return ','.join([sub for sub in chn_subset])


def _get_serial_number_from_meta(md):
    """
    Get neuropixel serial number from the metadata dictionary
    """
    # imProbeSN for 3A, imDatPrb_sn for 3B2, None for nidq 3B2
    serial = md.get('imProbeSN') or md.get('imDatPrb_sn')
    if serial:
        return int(serial)


def _get_neuropixel_major_version_from_meta(md):
    MAJOR_VERSION = {'3A': 1, '3B2': 1, '3B1': 1, 'NP2.1': 2, 'NP2.4': 2.4}
    version = _get_neuropixel_version_from_meta(md)
    if version is not None:
        return MAJOR_VERSION[version]


def _get_neuropixel_version_from_meta(md):
    """
    Get neuropixel version tag (3A, 3B1, 3B2) from the metadata dictionary
    """
    if 'typeEnabled' in md.keys():
        return '3A'
    prb_type = md.get('imDatPrb_type')
    # Neuropixel 1.0 either 3B1 or 3B2 (ask Olivier about 3B1)
    if prb_type == 0:
        if 'imDatPrb_port' in md.keys() and 'imDatPrb_slot' in md.keys():
            return '3B2'
        else:
            return '3B1'
    # Neuropixel 2.0 single shank
    if prb_type == 21:
        return 'NP2.1'
    # Neuropixel 2.0 four shank
    if prb_type == 24:
        return 'NP2.4'


def _get_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    ntr = int(_get_nchannels_from_meta(md))
    if typ == 'nidq':
        nsync = int(md.get('snsMnMaXaDw')[-1])
    elif typ in ['lf', 'ap']:
        nsync = int(md.get('snsApLfSy')[2])
    return list(range(ntr - nsync, ntr))


def _get_analog_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    if typ != 'nidq':
        return []
    tr = md.get('snsMnMaXaDw')
    nsa = int(tr[-2])
    return list(range(int(sum(tr[0:2])), int(sum(tr[0:2])) + nsa))


def _get_nchannels_from_meta(md):
    return int(md.get('nSavedChans'))


def _get_fs_from_meta(md):
    if md.get('typeThis') == 'imec':
        return md.get('imSampRate')
    else:
        return md.get('niSampRate')


def _get_type_from_meta(md):
    """
    Get neuropixel data type (ap, lf or nidq) from metadata
    """
    snsApLfSy = md.get('snsApLfSy', [-1, -1, -1])
    if snsApLfSy[0] == 0 and snsApLfSy[1] != 0:
        return 'lf'
    elif snsApLfSy[0] != 0 and snsApLfSy[1] == 0:
        return 'ap'
    elif snsApLfSy == [-1, -1, -1] and md.get('typeThis', None) == 'nidq':
        return 'nidq'


def _geometry_from_meta(meta_data):
    """
    Gets the geometry, ie. the full trace header for the recording
    :param meta_data: meta_data dictionary as read by ibllib.io.spikeglx.read_meta_data
    :return: dictionary with keys 'row', 'col', 'ind', 'shank', 'adc', 'x', 'y', 'sample_shift'
    """
    cm = _map_channels_from_meta(meta_data)
    major_version = _get_neuropixel_major_version_from_meta(meta_data)
    if cm is None:
        th = neuropixel.trace_header(version=major_version)
        th['flag'] = th['x'] * 0 + 1.
        return th
    th = cm.copy()
    if major_version == 1:
        # the spike sorting channel maps have a flipped version of the channel map
        th['col'] = - cm['col'] * 2 + 2 + np.mod(cm['row'], 2)
    th.update(neuropixel.rc2xy(th['row'], th['col'], version=major_version))
    th['sample_shift'], th['adc'] = neuropixel.adc_shifts(version=major_version)
    th['ind'] = np.arange(cm['col'].size)
    return th


def _map_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of channel positions along the shank

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: dictionary of arrays 'shank', 'col', 'row', 'flag', one value per active site
    """
    if 'snsShankMap' in meta_data.keys():
        chmap = re.findall(r'([0-9]*:[0-9]*:[0-9]*:[0-9]*)', meta_data['snsShankMap'])
        # for digital nidq types, the key exists but does not contain any information
        if not chmap:
            return {'shank': None, 'col': None, 'row': None, 'flag': None}
        # shank#, col#, row#, drawflag
        # (nb: drawflag is one should be drawn and considered spatial average)
        chmap = np.array([np.float32(cm.split(':')) for cm in chmap])
        return {k: chmap[:, v] for (k, v) in {'shank': 0, 'col': 1, 'row': 2, 'flag': 3}.items()}


def _conversion_sample2v_from_meta(meta_data):
    """
    Interpret the meta data to extract an array of conversion factors for each channel
    so the output data is in Volts
    Conversion factor is: int2volt / channelGain
    For Lf/Ap interpret the gain string from metadata
    For Nidq, repmat the gains from the trace counts in `snsMnMaXaDw`

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: numpy array with one gain value per channel
    """

    def int2volts(md):
        """ :return: Conversion scalar to Volts. Needs to be combined with channel gains """
        if md.get('typeThis', None) == 'imec':
            if 'imMaxInt' in md:
                return md.get('imAiRangeMax') / int(md['imMaxInt'])
            else:
                return md.get('imAiRangeMax') / 512
        else:
            return md.get('niAiRangeMax') / 32768

    int2volt = int2volts(meta_data)
    version = _get_neuropixel_version_from_meta(meta_data)
    # interprets the gain value from the metadata header:
    if 'imroTbl' in meta_data.keys():  # binary from the probes: ap or lf
        sy_gain = np.ones(int(meta_data['snsApLfSy'][-1]), dtype=np.float32)
        # imroTbl has 384 entries regardless of no of channels saved, so need to index by n_ch
        # TODO need to look at snsSaveChanMap and index channels to get correct gain
        n_chn = _get_nchannels_from_meta(meta_data) - len(_get_sync_trace_indices_from_meta(meta_data))
        if 'NP2' in version:
            # NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            out = {'lf': np.hstack((int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain)),
                   'ap': np.hstack((int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain))}
        else:
            # the sync traces are not included in the gain values, so are included for
            # broadcast ops
            gain = re.findall(r'([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)',
                              meta_data['imroTbl'])[:n_chn]
            out = {'lf': np.hstack((np.array([1 / np.float32(g.split(' ')[-1]) for g in gain]) *
                                    int2volt, sy_gain)),
                   'ap': np.hstack((np.array([1 / np.float32(g.split(' ')[-2]) for g in gain]) *
                                    int2volt, sy_gain))}

    # nidaq gain can be read in the same way regardless of NP1.0 or NP2.0
    elif 'niMNGain' in meta_data.keys():  # binary from nidq
        gain = np.r_[
            np.ones(int(meta_data['snsMnMaXaDw'][0], )) / meta_data['niMNGain'] * int2volt,
            np.ones(int(meta_data['snsMnMaXaDw'][1], )) / meta_data['niMAGain'] * int2volt,
            np.ones(int(meta_data['snsMnMaXaDw'][2], )) * int2volt,  # no gain for analog sync
            np.ones(int(np.sum(meta_data['snsMnMaXaDw'][3]), ))]  # no unit for digital sync
        out = {'nidq': gain}

    return out


def split_sync(sync_tr):
    """
    The synchronization channels are stored as single bits, this will split the int16 original
    channel into 16 single bits channels

    :param sync_tr: numpy vector: samples of synchronisation trace
    :return: int8 numpy array of 16 channels, 1 column per sync trace
    """
    sync_tr = np.int16(np.copy(sync_tr))
    out = np.unpackbits(sync_tr.view(np.uint8)).reshape(sync_tr.size, 16)
    out = np.flip(np.roll(out, 8, axis=1), axis=1)
    return np.int8(out)


def get_neuropixel_version_from_folder(session_path):
    ephys_files = glob_ephys_files(session_path, ext='meta')
    return get_neuropixel_version_from_files(ephys_files)


def get_neuropixel_version_from_files(ephys_files):
    if any([ef.get('nidq') for ef in ephys_files]):
        return '3B'
    else:
        return '3A'

def _mock_spikeglx_file(mock_bin_file, meta_file, ns, nc, sync_depth,
                        random=False, int2volts=0.6 / 32768, corrupt=False):
    """
    For testing purposes, create a binary file with sync pulses to test reading and extraction
    """
    meta_file = Path(meta_file)
    mock_path_bin = Path(mock_bin_file)
    mock_path_meta = mock_path_bin.with_suffix('.meta')
    md = read_meta_data(meta_file)
    assert meta_file != mock_path_meta
    fs = _get_fs_from_meta(md)
    fid_source = open(meta_file)
    fid_target = open(mock_path_meta, 'w+')
    line = fid_source.readline()
    while line:
        line = fid_source.readline()
        if line.startswith('fileSizeBytes'):
            line = f'fileSizeBytes={ns * nc * 2}\n'
        if line.startswith('fileTimeSecs'):
            if corrupt:
                line = f'fileTimeSecs={ns / fs + 1.8324}\n'
            else:
                line = f'fileTimeSecs={ns / fs}\n'
        fid_target.write(line)
    fid_source.close()
    fid_target.close()
    if random:
        D = np.random.randint(-32767, 32767, size=(ns, nc), dtype=np.int16)
    else:  # each channel as an int of chn + 1
        D = np.tile(np.int16((np.arange(nc) + 1) / int2volts), (ns, 1))
        D[0:16, :] = 0
    # the last channel is the sync that we fill with
    sync = np.int16(2 ** np.float32(np.arange(-1, sync_depth)))
    D[:, -1] = 0
    D[:sync.size, -1] = sync
    with open(mock_path_bin, 'w+') as fid:
        D.tofile(fid)
    return {'bin_file': mock_path_bin, 'ns': ns, 'nc': nc, 'sync_depth': sync_depth, 'D': D}

