import os
import numpy as np
from collections import defaultdict
import forgi.graph.bulge_graph as fgb
import forgi
import RNA

## FILE PATH UTILS ###

def getFileList(path, ftype):
    file_list = []    
    filenames = os.listdir(path)
    file_extensions = set(['.{}'.format(ftype)])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(path, filename)
            file_list.append(full_filename)
    return file_list

### ENCODING UTILS ###

def onehotTableRNA():
    onehot = defaultdict(lambda: np.array([.25,.25,.25,.25]))
    onehot["A"] = np.array([1,0,0,0])
    onehot["C"] = np.array([0,1,0,0])
    onehot["G"] = np.array([0,0,1,0])
    onehot["U"] = np.array([0,0,0,1])
    return onehot


def onehotTableDot():
    onehot_dot = defaultdict(lambda: np.array([0,0,0,0,0,0,0,1]))
    onehot_dot["("] = np.array([1,0,0,0,0,0,0,0])
    onehot_dot[")"] = np.array([0,0,0,0,0,0,1,0])
    onehot_dot["."] = np.array([0,0,0,1,0,0,0,0])
    onehot_dot["["] = np.array([0,1,0,0,0,0,0,0])
    onehot_dot["]"] = np.array([0,0,0,0,0,1,0,0])
    onehot_dot["{"] = np.array([0,0,1,0,0,0,0,0])
    onehot_dot["}"] = np.array([0,0,0,0,1,0,0,0])
    return onehot_dot

def elements_structure_encoding(elements_struct):
    onehot = defaultdict(lambda: np.array([0.1666,0.1666,0.1666,0.1666,0.1666,0.1666]))
    onehot["F"] = np.array([1,0,0,0,0,0])
    onehot["T"] = np.array([0,1,0,0,0,0])
    onehot["S"] = np.array([0,0,1,0,0,0])
    onehot["I"] = np.array([0,0,0,1,0,0])
    onehot["M"] = np.array([0,0,0,0,1,0])
    onehot["H"] = np.array([0,0,0,0,0,1])
    
    onehot_ess = [onehot[ss] for ss in elements_struct]
    return np.array(onehot_ess)

### RNA SECONDARY STRUCTURE UTILS ###

def get_RNA_secondary_structure_dot_bracket(seq):
    ss, _ = RNA.fold(seq)
    return ss

def get_RNA_secondary_structure_free_energy(seq):
    _ , mfe = RNA.fold(seq)
    return mfe
    
def save_RNA_secondary_structure_svg(seq, ss, file_name):
    RNA.svg_rna_plot(seq, ss, file_name)
    return 0

def dotbracket_to_elements(dotbracket_struct):
    bg = fgb.BulgeGraph.from_dotbracket(dotbracket_struct)
    es = fgb.BulgeGraph.to_element_string(bg, with_numbers=False)
    es = es.upper()            
    return es

