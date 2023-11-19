import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import os
import pickle
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap

from anatomy import *
from lfp import *

## Set this to histology export path for convenience ##
## Alternatively pass a path to "load_histology()" ##
HISTOLOGY_PATH = "/path/to/histology/neuropixel/export"

class MultiprobeEphysExperiment(object):
    """
    Object representing a combined recording with behavior, histology, and spiking data for a single recording session.
    
    This object can contain multiple simultaneously recorded sorted datasets + multiple 
    behavioral sessions. See jupyter notebook for example usage.

    For convenience, this class may be sub-classed to add experiment-specific
    behavioral data or event processing, such as trials and other event triggers.

    """
    def __init__(self, penetration=None, recording=None, mouse_name=None):
        self._recording = None
        self._behavior = [] # list of SessionBehavior objects containing behavior information for each session
        self._video = [] # VideoData objects with PCs representing animal motion during behavior, for each session.
        self._histology = None
        self._unit_data = pd.DataFrame()
        self._units = None
        self._nrecording = 0

        # database values
        self._penetration = penetration
        self._recording = recording
        self._mouse_name = mouse_name
        if penetration is not None:
            self._tract_names = penetration.split(';')
        else:
            self._tract_names = []

        self._unit_locs = None
        self._unit_areas = None
        self._unit_area_ids = None

        self._atlas = None

    def load_histology(self, atlas, histology_path=None):

        if histology_path is None:
            histology_path = HISTOLOGY_PATH
        
        fname = os.path.join(histology_path,
                    self._mouse_name + "_" + self._recording + ".npz")
        if os.path.exists(fname):
            S = np.load(fname)
            self._atlas = atlas
            self._unit_locs = S["transformed_locs"]
            self._chan_locs = S["transformed_chans"]
            self._chan_probe_id = S["chan_probe_id"]
        else:
            print('Path not found: %s' % fname )

    def load_units_from_ks(self, fpath, verbose=True, load_lfp=False):
        """
        Load units from numpy array, then compute firing rates.
        NOTE: The KS data should have been aligned to sync events, so we can
        assume that everything is in the same timeframe.
        """
        if ".npz" in fpath:
            data = np.load(fpath, allow_pickle=True)
        else:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
        
        self._spike_times = data._spike_times
        self._clusts = data._clusts
        self._clust_id = data._clust_id
        self._clust_depths = data._clust_depths
        self._probe_id = data._probe_id
        self._metrics = data._metrics

        if load_lfp:
            self._lfp = data._lfp
            self._lfp_fs = data._lfp_fs
            self._lfp_chans = data._lfp_chans
            self._lfp_probe_ids = data._lfp_probe_ids
        else:
            self._lfp = None
            self._lfp_fs = None
            self._lfp_chans = None
            self._lfp_probe_ids = None

        # convenience data for performant spike extraction
        self.__clusts_argsorted = np.argsort(self._clusts)
        self.__clusts_sorted = self._clusts[self.__clusts_argsorted]
        # event sync data
        self._events = data._events

        self._ntot = data._ncell
        self._max_t = self._spike_times.max()
        self.get_good_units()


    def get_good_units(self, isi_viol=0.1, snr=1.5, nspikes=500, noise=0, qc='bombcell'):
        m = self._metrics
        if qc=='bombcell':
            good_annot = m["good"].astype('bool')
        else:
            good_annot = m["noise"]==noise
        mask = (m["isi_viol"]<isi_viol) & (m["snr"]>=1.5) & (m["n_spikes"]>nspikes) & good_annot
        self._good = np.unique(m[mask]["cluster_id"])
        self._ncell = len(self._good)
        self._good_mask = mask

    def get_expt_name(self):
        return self._penetration + "_" + self._mouse_name + "_" + self._recording

    def get_ncells(self):
        return self._ncell
 
    def get_clust_ids(self):
        return self._clust_id

    def get_clust_groups(self):
        """
        Return cluster groups: good, MUA, noise
        """
        return self._clust_groups

    def get_maxt(self):
        return self._max_t

    def get_lfp_for_probe(self,probe_id):
        return self._lfp[:, self._lfp_probe_ids==probe_id]

    #
    # METHODS RELATED TO ANATOMY
    #

    def get_brain_area_map(self, use_higher=False, locs=None):
        mask = True if locs is None else False
        ids = self.get_brain_areas(info='id', mask=mask, locs=locs)
        acronyms = self.get_brain_areas(info='acronym', mask=mask, locs=locs)
        name_to_id = {}
        id_to_name = {}
        for i,a in enumerate(acronyms):
            name_to_id[a] = ids[i]
            id_to_name[ids[i]] = a
        return name_to_id, id_to_name

    def get_brain_areas(self, info='id', level='bottom', mask=True, locs=None):
        if locs is None:
            locs = self._unit_locs
        # NOTE THAT THESE ARE SORTED BY DEPTH IN REVERSE ORDER!!
        areas = self._atlas.get_info_for_points(locs, element="id", increase_level=False)
        areas = np.array([self._atlas.get_acronym(a, level=level) for a in areas])
        if info == 'id':
            # doing this a second time to get the area id at the appropriate "level"
            areas = np.array([-1 if ar=='NA' else self._atlas._tree.get_id_acronym_map()[ar] for ar in areas])
        if mask:
            return areas[self._good_mask]
        else:
            return areas

    def get_areas_sorted_hierarchically(self, locs=None):
        """ Get index of good units sorted hierarchically by brain region subdivision
        """
        mask = True if locs is None else False
        area_ids = self.get_brain_areas(info='id', level='bottom', mask=mask, locs=locs)
        area_ix = np.arange(len(area_ids))
        areas_top = self.get_brain_areas(info='acronym', level='top', mask=mask, locs=locs)
        area_hierarchy_sorted = np.hstack([area_ix[areas_top==region][np.argsort(area_ids[areas_top==region])] for region in np.unique(areas_top)])
        return area_hierarchy_sorted

    def get_area_colors(self, area_ids):
        """
        Get Allen CCF colors of areas by id, with unassigned areas mapped to white.

        Example usage: plt.plot(regional_psth, color=get_area_colors([region_id]))
        """
        area_colors = np.array([(255,255,255) if aid==-1 else self._atlas._tree.get_colormap()[aid] for aid in area_ids])/255
        return area_colors

    def get_area_colorbar(self, area_ids):
        """
        Given a list of area ids, generate a colorbar (for plotting by imshow)
        and an associated colormap. Uses colors from the Allen CCF.

        Example usage: plt.imshow(colorbar, cmap=cm, aspect='auto')
        Example usage: plt.imshow(colorbar[sorted_idx,:], cmap=cm, aspect='auto')
        """
        id_color = []
        distinct_color = []
        for i,aid in enumerate(area_ids):
            if aid not in id_color:
                id_color.append(aid)
                if aid == -1:
                    distinct_color.append(np.array([1.,1.,1.]))
                else:
                    distinct_color.append(np.array(self._atlas._tree.get_colormap()[aid])/255)
        id_color = dict(zip(id_color, range(len(id_color))))

        colors = np.pad(np.array(distinct_color), [0,1], constant_values=1)[:-1,:]
        cm = ListedColormap(colors)
        colorbar = np.expand_dims([id_color[aid] for aid in area_ids],1)
        return colorbar, cm

    def get_locs_for_lfp(self):
        return np.concatenate([self._chan_locs[self._chan_probe_id==p][self._lfp_chans[self._lfp_probe_ids==p].astype('int')] for p in np.unique(self._chan_probe_id)])

    #    
    # METHODS RELATING TO GETTING SPIKES
    # 
    def firing_rates_over_recording(self,bin_size=0.01, 
        smooth_size=None):
        """
        Return a matrix containing the binned firing rate for each neurons 
        over the whole recording session.
        Returns the matrix and an array of the start time of each bin of 
        the matrix.
        :input bin_size: bin size in seconds
        """
        nbins = self._max_t/float(bin_size)
        t = np.arange(0, self._max_t, bin_size)
        fr = np.zeros((len(self._good), len(t)-1)) 
        for idx,clu in enumerate(self._good):
            curr_spikes = self.get_spikes_for_clust(clu).flatten()
            # array of ones of the same size as the spike times, 
            # to compute the count
            t = np.ones_like(curr_spikes) 
            curr_fr,bin_edges,_ = binned_statistic(curr_spikes, t, 
                statistic="sum", bins=nbins, range=(0, self._max_t))
            fr[idx,:] = curr_fr/bin_size
            if smooth_size:
                fr[idx,:] = moving_avg_filter(fr[idx,:], smooth_size)

        return fr, bin_edges

    
    def get_spikes_per_event(self, event_times,fwd_t, rev_t=0, shift=True, use_both=False):
        """
        Return list of nCell of list of nEvent of spikes per event
        Args:
            event_times: time in seconds of each event to obtain spikes from around
            fwd_t (float): length of event in seconds
            rev_t (float, optional): preceding time in seconds
            shift (bool, optional): whether to shift time of each spike relative to the event time. Default is True. NOTE: Corrected for rev_t so min spike time is 0, max is rev_t+fwd_t
        Returns:
            spikes_per_cell: list of spikes per event_times, per cell (Ncell list of Ntrials)
        """
        if use_both:
            cell_ids = np.hstack((self._good, self._mua))
        else:
            cell_ids = self._good
        spikes_per_cell = [None for i in range(len(cell_ids))]
        for i in range(len(spikes_per_cell)):
            spikes_per_cell[i] = [None]*len(event_times)
        # only operate on "good" cells for now (ignore MUA)
        for i,idx in enumerate(cell_ids):
            spikes_for_clust = np.sort(self.get_spikes_for_clust(idx))
            for t,s in enumerate(event_times):
                start,stop = np.searchsorted(spikes_for_clust,[s-rev_t, s+fwd_t])
                spikes_per_cell[i][t] = spikes_for_clust[start:stop]
                #spikes_per_cell[i][t] = spikes_for_clust[np.logical_and(spikes_for_clust >= (s - rev_t), spikes_for_clust < (s + fwd_t))] 
                if shift:
                    spikes_per_cell[i][t] -= s
                    spikes_per_cell[i][t] += rev_t # make spikes start at 0
        return spikes_per_cell
 
    # METHODS RELATED TO SPATIAL LOCATION ON PROBE
    #
    def get_clusters_sorted_by_depth(self, clust_type="good"):
        """
        Computes good units sorted by depth along electrode.
        Returns:
             cluster_id: sorted list of ids of clusters
             sorted_idx: indices of each cluster (e.g. to rearrange firing rate matrix)
        """
        if clust_type == "both":
            sorted_idx = np.argsort(self._clust_depths)
        elif clust_type == "mua":
            raise NotImplementedError
            #sorted_idx = np.argsort(self._clust_depths[self._clust_groups==1])
        elif clust_type == "good":
            sorted_idx = np.argsort(self._clust_depths[self._good])
        return self._clust_id[sorted_idx], sorted_idx
    
    def depth_sort_probe(self, probe):
        return self._clust_depths[(data._good) & (data._probe_id==probe)].argsort()

    def get_cluster_depths(self, clust_type="good"):
        if clust_type == "both":
            return self._clust_depths
        elif clust_type == "mua":
            raise NotImplementedError
            #return self._clust_depths[self._clust_groups==1]
        elif clust_type == "good":
            return self._clust_depths[self._good]

    #
    def get_spikes_for_clust(self,clust_id,use_timestamps=False):
        """
        Return timestamps for spikes from a particular cluster, in s
        """
        b, e = np.searchsorted(self.__clusts_sorted, [clust_id, clust_id+1])
        return self._spike_times[self.__clusts_argsorted[b:e]]
        #return self._spike_times[self._clusts == clust_id]

    def get_nearest_event(self, events, t):
        return events[np.argmin(np.abs(events-t))]

# Utilities

def moving_avg_filter(t,win_size=10,causal=True):
    if causal:
        filtered = np.append(t[:win_size-1], pd.Series(t).rolling(window=win_size).mean().iloc[win_size-1:].values)
    else:
        filtered = np.append(pd.Series(t).rolling(window=win_size).mean().iloc[win_size-1:].values, t[-win_size+1:])
    assert t.size==filtered.size
    return filtered

def moving_avg_filter_2d(fr, win_size=10, causal=True):
    assert len(fr.shape)==2
    return np.vstack(Parallel(n_jobs=-1)(delayed(moving_avg_filter)(fr[i,:], win_size=win_size, causal=causal) for i in np.arange(fr.shape[0])))

def moving_avg_filter_3d(fr, win_size=10, causal=True):
    assert len(fr.shape)==3
    return np.stack(Parallel(n_jobs=-1)(delayed(moving_avg_filter_2d)(fr[:,i,:], win_size=win_size, causal=causal) for i in np.arange(fr.shape[1])),1)

def find(x):
    return np.argwhere(x).flatten()