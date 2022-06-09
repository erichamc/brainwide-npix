# This code loads Neuropixels data acquired by SpikeGLX and processed by:
# CatGT, Kilosort, Tprime, CWaves, and associated postprocessing from https://github.com/jenniferColonell/ecephys_spike_sorting
# Data is packaged into a numpy object providing convenient access to spike times, event times, and relevant cluster metrics

import numpy as np
import os
import sys
import glob
import pandas as pd
import pickle
import scipy
from joblib import Parallel, delayed

import lfp
import sglx_util


class MultiprobeData(object):
    """
    Class used to load all the sorting results from a single experiment. 
    This assumes that Tprime has already been run and aligned event times are available. 
    """
    def __init__(self, mouse, date, load_waveforms=False, load_lfp=True):
        self._expts = [] # list of probe paths
        self._ncell = 0
        self._spike_times = None # Nspikes x 1
        self._clusts = None # Nspikes x 1
        self._clust_id = None # Nclust x 1 IDs of each cluster (in order)
        self._clust_depths = None
        self._clust_amps = None
        self._probe_id = None # Nclust x 1 which probe each cluster came from
        self._tract_names = []
        self._chan_map = []
        self._chan_pos = []
        self._templates = []
        self._winv = []
        self._mean_waveforms = None
        self._ntemplates = 0
        self._spike_templates = []
        self._mouse = mouse
        self._date = date

        self._lfp = []
        self._lfp_fs = []
        self._lfp_chans = []
        self._lfp_probe_ids = []

        self._load_lfp = load_lfp
        self._load_waveforms = load_waveforms
        # experiment-specific event info
        self._events = None

    def add_experiment(self, probe_path, tract_name):
        # Add probe paths
        self._expts.append(probe_path)
        self._tract_names.append(tract_name)

    def add_events(self, data_path, event_map):
        # Customize based on recorded events on the Nidaq
        # Assume already aligned using Tprime
        # data_path: path to folder containing Tprime output
        # event_map: name of events corresponding to nidaq channels
        # note that the key in event_map should unambiguously match substring in the tprime output file name.
        # For example: {'XA_2': 'trial', 'XA_3': 'stim'}
        events = {}
        for k,v in event_map.items():
            path = glob.glob(os.path.join(data_path, f"*{k}*"))[0]
            events[v] = np.loadtxt(path)
        self._events = events


    def combine_experiments(self, ref_expt_idx=0, clust_offset=10000):
        """
        Inputs:
            ref_expt_idx: experiment index to use as reference
            clust_offset: maximum number of possible clusters in each experiment, to uniquely identify clusters
        """
        for i, probe_path in enumerate(self._expts):
            expt = EPhyClusts(self._expts[i], self._load_waveforms, load_lfp=self._load_lfp)
            self._chan_map.append(expt._chan_map)
            self._chan_pos.append(expt._chan_pos)
            self._winv.append(expt._winv)
            self._templates.append(expt._templates)
            self._spike_templates.append(expt._spike_templates)
            self._ntemplates += expt._ntemplates
            self._ncell += len(expt._metrics[expt._metrics["noise"]==0])
            #self._amplitudes.append(expt._amplitudes)
            if self._load_waveforms:
                if i == 0:
                    self._mean_waveforms = expt._mean_waveforms
                else:
                    self._mean_waveforms = np.concatenate([self._mean_waveforms, expt._mean_waveforms])
            
            old_clust_ids = expt._metrics["cluster_id"] # id numbers
            old_clusts = expt._clusts # spikes tagged by id
            new_clusts = old_clusts.copy()
            new_clust_ids = old_clust_ids + clust_offset*i
            expt._metrics["n_spikes"] = np.unique(old_clusts, return_counts=True)[1]
            # Performant relabeling of the spikes
            reid_dict = dict(zip(old_clust_ids, new_clust_ids))
            replace = np.array([list(reid_dict.keys()), list(reid_dict.values())])
            new_clusts = replace[1, np.searchsorted(replace[0, :], old_clusts)]
            expt._metrics["cluster_id"] = new_clust_ids

            if i == 0:
                self._clust_depths = expt._clust_depths
                self._clust_xpos = expt._clust_xpos
                self._probe_id = np.zeros(len(expt._clust_id),dtype=np.int)
                self._spike_times = expt._spike_times
                self._clust_id = new_clust_ids
                self._clusts = new_clusts
                self._metrics = expt._metrics
                if self._load_lfp:
                    self._lfp.append(expt._lfp)
                    self._lfp_fs.append(expt._lfp_fs)
                    self._lfp_chans = expt._lfp_chans
                    self._lfp_probe_ids = i*np.ones(len(expt._lfp_chans), dtype=np.int)
            else:
                self._clust_depths = np.concatenate([self._clust_depths, expt._clust_depths])
                self._clust_xpos = np.concatenate([self._clust_xpos, expt._clust_xpos])
                self._probe_id = np.concatenate([self._probe_id, i*np.ones(len(expt._clust_id), dtype=np.int)])
                self._spike_times = np.concatenate([self._spike_times, expt._spike_times])
                self._clust_id = np.concatenate([self._clust_id, new_clust_ids])
                self._clusts = np.concatenate([self._clusts, new_clusts])
                self._metrics = pd.concat([self._metrics, expt._metrics], ignore_index=True)
                if self._load_lfp:
                    self._lfp.append(expt._lfp)
                    self._lfp_fs.append(expt._lfp_fs)
                    self._lfp_chans = np.concatenate([self._lfp_chans, expt._lfp_chans])
                    self._lfp_probe_ids = np.concatenate([self._lfp_probe_ids, i*np.ones(len(expt._lfp_chans), dtype=np.int)])
        
        print("Sorting spike times")
        sort_idx = np.argsort(self._spike_times)
        
        self._clusts = self._clusts[sort_idx]
        self._spike_times = self._spike_times[sort_idx]

        if self._load_lfp:
            self._lfp, self._lfp_fs = self.rescale_lfps(self._lfp, self._lfp_fs)

    
    def save_data(self, out_path):
        with open(out_path, 'wb') as f:
            pickle.dump(self,f)

    def save_units_npz(self, out_path):
        np.savez(out_path, spike_times=self._spike_times, clusts=self._clusts,
            clust_id=self._clust_id, clust_depths=self._clust_depths,
            probe_id=self._probe_id, tract_names=self._tract_names, 
            metrics=self._metrics, clust_xpos=self._clust_xpos, 
            chan_map=self._chan_map, chan_pos=self._chan_pos, winv=self._winv,
            templates=self._templates, spike_templates=self._spike_templates, 
            ncell=self._ncell, mean_waveforms=self._mean_waveforms,
            events=self._events, mouse=self._mouse, date=self._date)

    def load_units(self, fpath):
        pass

    def rescale_lfps(self, lfps, sample_rates):
        """
        Take in a list of LFP data and a list of sample rates.
        Resize LFP data in time dimension to slowest sampling rate.
        Return the scaled LFPs and the new (minimum) sampling rate
        """
        min_fs_ix = np.argmin(sample_rates)
        min_fs = sample_rates[min_fs_ix]
        zoom_factors = [(min_fs/fs,1) for fs in sample_rates]

        scaled_lfps = Parallel(n_jobs=-1)(delayed(scipy.ndimage.zoom)(lfps[i], zoom_factors[i], order=1) for i in range(len(sample_rates)))
        shapes = np.array([l.shape[0] for l in scaled_lfps])
        scaled_lfps = [l[:np.min(shapes),:] for l in scaled_lfps]
        scaled_lfps = np.concatenate(scaled_lfps, axis=1)
        return scaled_lfps, min_fs

class EPhyClusts(object):
    """
    Class to load the results of a KiloSort + Phy + Quality Scoring + CWaves for a single probe. 

    This class is used to process condense the Phy output into a single npy file.
    """

    def __init__(self, dirname, load_waveforms=False, load_lfp=True):
        """
        :input dirname: name of kilosort output directory with Phy data
.
        """
        vals = {}
        # load params from python file
        print("Loading ephys variables")
        with open(os.path.join(dirname, "params.py")) as f:
            for line in f:
                fields = line.rstrip().split(" ")
                vals[fields[0]] = fields[2]
        self._dirname = dirname
        self._dat_path = vals['dat_path'][1:-1] # to unquote
        self._n_channels_dat = vals['n_channels_dat']
        self._dtype = vals['dtype']
        self._offset = vals['offset']
        #self._sample_rate = float(vals['sample_rate'])
        self._hp_filtered = vals['hp_filtered']

        # load all variables
        self._chan_map = np.load(os.path.join(dirname, "channel_map.npy")) # active channels
        self._chan_pos = np.load(os.path.join(dirname, "channel_positions.npy")) # channel positions in um 
        self._spike_times = np.load(os.path.join(dirname, "spike_times_sec_adj.npy")).flatten() # time adjusted spikes in seconds
        if load_waveforms:
            self._mean_waveforms = np.load(os.path.join(dirname, "mean_waveforms.npy")) # average waveforms for each cluster. this is hundreds of MBs
        else:
            self._mean_waveforms = None
        
        self._clusts = np.load(os.path.join(dirname, "spike_clusters.npy")).flatten() # cluster for each spike
        self._spike_templates = np.load(os.path.join(dirname, "spike_templates.npy")).flatten()
        self._templates = np.load(os.path.join(dirname, "templates.npy")) # the whitened template waveforms [nTemplates x nTimesPoints x nChannels]
        self._ntemplates = self._templates.shape[0]
        self._winv = np.load(os.path.join(dirname, "whitening_mat_inv.npy")) # used to unwhiten templates into raw data space

        if os.path.exists(os.path.join(dirname, "cluster_info.tsv")):
            self._metrics = pd.read_csv(os.path.join(dirname, "cluster_info.tsv"), sep="\t")
        else:
            self._metrics = pd.read_csv(os.path.join(dirname, "metrics.csv"))
        _, self._noise, _ = self._load_cluster_types(dirname)
        self._metrics["noise"] = np.zeros_like(self._metrics["cluster_id"])
        self._metrics.loc[self._metrics["cluster_id"].isin(self._noise), "noise"] = 1
        self._clust_id = np.unique(self._metrics["cluster_id"])
        self._clust_depths = self._chan_pos[self._metrics["peak_channel"][np.argsort(self._metrics["cluster_id"])]][:,1]
        self._clust_xpos = self._chan_pos[self._metrics["peak_channel"][np.argsort(self._metrics["cluster_id"])]][:,0]
        
        if load_lfp:
            self._lfp, self._lfp_chans, self._lfp_fs = self.load_downsampled_lfp(dirname)
        else:
            self._lfp, self._lfp_chans, self._lfp_fs = (None, None, None)

    def load_downsampled_lfp(self, dirname, temporal_downsample=10, spatial_downsample=4):
        """
        Load and downsample LFP for a given probe.
        """
        parent_dir = os.path.dirname(dirname)
        lf_fn = glob.glob(os.path.join(parent_dir, f"*imec*.lf.bin"))[0]
        chanmap_fn = glob.glob(os.path.join(parent_dir, "*chanMap.mat"))[0]
        connected = np.squeeze(scipy.io.loadmat(chanmap_fn)["connected"])
        lfp_data = sglx_util.Reader(lf_fn)
        fs = lfp_data.fs
        # subsample LFP data spatially and temporally -- take every 4th channel, every 10 samples
        lfp_sub = lfp.subsample_lfp(lfp_data._raw, np.arange(384)[connected.astype('bool')][::spatial_downsample], temporal_downsample)
        chan_ix = np.arange(np.sum(connected))[::spatial_downsample]
        lfp_data.close()
        return lfp_sub, chan_ix, fs/temporal_downsample
    
    def _load_cluster_types(self, dirname):
        noise = []
        good = []
        mua = []
        fname = os.path.join(dirname, "cluster_group.tsv")
        with open(fname) as f:
            for l in f:
                fields = l.rstrip().split("\t")
                if fields[0] == "cluster_id":
                    pass
                else:
                    cluster_id = int(fields[0])
                    cluster_type = fields[1]
                    if cluster_type == "good":
                        good.append(cluster_id)
                    elif cluster_type == "mua":
                        mua.append(cluster_id)
                    elif cluster_type == "noise":
                        noise.append(cluster_id)
        return np.array(good), np.array(noise), np.array(mua)