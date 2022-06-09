
import os
import numpy as np
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.config.manifest import Manifest
from allensdk.core.reference_space import ReferenceSpace
import pandas as pd

# This could get packaged locally into a git repo but now is a link to
# an external allen ccf annotation volume directory
ANNOTATION_DIR = "/path/to/reference/ccf"

def get_idx_for_areas(vals):
    """
    Given a list of the brain area names or ids per some set of points, return the unique set of names/ids mapped to the indices
    that comes from each name.
    Use case: get the indices of all points that come from each of a set of brain areas.
    Args:
        areas: Npoint long list of brain area acronyms,names, or ids
    Returns:
        idx_map: map from unique acronyms,names,ids to indices in original list that have those values.
    
    """
    idx_map = {}
    for i in np.unique(vals):
        idx = np.nonzero(np.array(vals)==i)[0]
        idx_map[str(i)] = np.array(idx)
    return idx_map

class AIBSAtlas(object):
    """
    Class encapsulating the AIBS atlas
    """
    def __init__(self, base_path=ANNOTATION_DIR, max_z=None):
        self._base_path = base_path
        self._load_tree()        
        self._load_annotation(max_z=max_z)
        self._create_ref_space()

        # cached points and labels
        self._cached_points = None
        self._cached_labels = None
        self._cached_nclusts = None

        self._allenccf_st = self._load_allenccf2017_tree()

        self._top_level = [
            "Isocortex",
            "OLF", # Olfactory areas
            "HPF", # Hippocampal formation
            "CTXsp", # Cortical subplate
            "STR", # Striatum
            "PAL", # Pallidum
            "TH", # Thalamus
            "HY", # Hypothalamus
            "MB", # Midbrain
            "HB", # Hindbrain
            "CB", # Cerebellum   
        ]
        self._hierarchy = {
            # Graph level to go to for "middle" level regions
            "Isocortex": -1,
            "OLF": -1,
            "HPF": -2,
            "CTXsp": -1,
            "STR": -1,
            "PAL": -1,
            "TH": -3,
            "HY": -2,
            "MB": -2,
            "HB": -3,
            "CB": -2,   
        }        

    def get_acronym(self, region_id, level="middle"):
        """
        Return the Allen Institute structure acronym by level.
        Levels can be top, middle, or bottom. Bottom is finest regional distinction possible.
        """
        ancestors = [r['acronym'] for r in self._tree.ancestors([region_id])[0]]
        for area in self._top_level:
            contained = area in ancestors
            if contained:
                ix = ancestors.index(area)
                # Take leaf if a given region is above the traversal depth
                idx = max(ix+self._hierarchy[area],0)
                sub_area = ancestors[idx]
                break
        if not contained:
            return "NA"
        if level=="bottom":
            # leaf node
            return ancestors[0]
        elif level=="middle":
            # target from structure hierarchy
            return sub_area
        elif level=="top":
            # matched by the break
            return area
        else:
            raise ValueError

    def _load_tree(self):
        print("Loading tree...")
        oapi = OntologiesApi()
        structure_graph = oapi.get_structures_with_sets([1])
        structure_graph = StructureTree.clean_structures(structure_graph)  
        self._tree = StructureTree(structure_graph)

    def _load_allenccf2017_tree(self):
        return pd.read_csv(ANNOTATION_DIR+'/'+'structure_tree_safe_2017.csv')

    def _get_id_from_ann(self, i):
        return self._allenccf_st.iloc[i-1]['id']

    def get_acronym_id_map(self, ids):
        uniq_ids = np.unique(ids)
        acronyms = self.get_acronym_for_ids(uniq_ids)
        id_to_acronym = {k:v for (k,v) in zip(uniq_ids, acronyms)}
        acronym_to_id = {k:v for (k,v) in zip(acronyms, uniq_ids)}
        return id_to_acronym, acronym_to_id

    def get_acronym_for_ids(self, ids):
        return [r['acronym'] for r in self._tree.get_structures_by_id(ids)]

    def _load_annotation(self,max_z=None):
        print("Loading annotation...")
        annotation_path = os.path.join(self._base_path, 'annotation_volume_10um_by_index.npy')
        template_path = os.path.join(self._base_path, 'template_volume_10um.npy')
        self._annotation = np.load(annotation_path)
        self._template = np.load(template_path)
        if max_z is not None:
            self._annotation = self._annotation[:,:,:max_z]
            self._template = self._template[:,:,:max_z]
        
    def _create_ref_space(self, spacing=[10,10,10]):
        print("Creating ref space...")
        self._rsp = ReferenceSpace(self._tree, self._annotation,spacing)
        self._rsp.remove_unassigned()


    def get_info_for_points(self, points, element='acronym', increase_level=False):
        """
        Args:
            points: Nx3 array of pixel coordinates of brain areas
            element: element from tree to recover. I.e. 'acronym' or 'id' or 'name'
            increase_level: get value for first ancestor in tree for each element
        Returns:
            vals: some 
            
        """
        vals = []
        for pos in np.round(points).astype(np.int):
            i = self._annotation[pos[0],pos[1],pos[2]]
            i = self._get_id_from_ann(i)
            curr_region = self._tree.get_structures_by_id([i])[0]
            if increase_level:
                # 0th element is curr_region, 1st is immediate ancestor, etc
                if curr_region is not None:
                    ancestor = self._tree.ancestor_ids([curr_region['id']])[0]
                    if len(ancestor) > 1:
                        curr_region = self._tree.get_structures_by_id([ancestor[1]])
            if curr_region is not None:
                if not isinstance(curr_region,list):
                    vals.append(curr_region[element])
                else:
                    vals.append(curr_region[0][element])
            else:
                if element == "acronym" or element == "name":
                    vals.append("NA")
                elif element == "id":
                    vals.append(-1)
        return vals