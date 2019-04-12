import os
from utils import *
from collections import defaultdict
import scipy.sparse as sp
import numpy as np

# Easy RNASTRAND database preprocessor 
# for Machine Learning, Deep Learning
class MLRNAStrandv2():
    def __init__(self):
        # static path
        self.npz = "preprocessed.npz"
        self.dp_path    = "RNA_STRAND_v2_0_dp"
        self.dp_type    = "dp"
        self.bpseq_path = "RNA_STRAND_v2_0_bpseq"
        self.bpseq_type = "bpseq"
        
    def load(self):
        dataset = np.load(self.npz)
        X    = dataset["X"] # X    : RNA sequence(onehot encoded)
        L    = dataset["L"] # L    : RNA sequence letter (not encoded)
        C    = dataset["C"] # C    : RNA sequence's secondary structure link info (dictionary)
        A    = dataset["A"] # A    : RNA sequence's secondary structure link adjacency matrix (scipy.sparse.coo_matrix)
        N    = dataset["N"] # N    : RNA sequence's length 
        S    = dataset["S"] # S    : RNA sequence's secondary structure dot-bracket format (onehot encoded)
        E    = dataset["E"] # E    : RNA sequence's secondary structure dot-bracket base structure elements (onehot encoded)
        ID   = dataset["ID"] # ids
        
        print("Load dataset : {}".format(self.npz))
        print("X : {}".format(X.shape))
        print("L : {}".format(L.shape))
        print("C : {}".format(C.shape))
        print("A : {}".format(A.shape))
        print("N : {}".format(N.shape))
        print("S : {}".format(S.shape))
        print("E : {}".format(E.shape))
        print("ID : {}".format(ID.shape))
        
        return X,L,C,A,N,S,E,ID
                
    def save(self):
        self.dp_file_list    = getFileList(self.dp_path, ftype=self.dp_type)
        self.bpseq_file_list = getFileList(self.bpseq_path, ftype=self.bpseq_type)
        
        # Table for RNA encoding
        self.onehot_RNA = onehotTableRNA()
        self.onehot_DOT = onehotTableDot()
        self.data_saver()
    
    def data_saver(self):
        # Get belows from bpseq file
        # 1. data-ID (ids)
        # 2. onehot-encoded sequence (X)
        # 3. seqeunce character (S)
        # 4. connection dictionary (C)
        # 5. adjacency matrix(sparse matrix) (A)
        # 6. sequence length (N)
                
        print("Start RNA dataset preprocessing...")
        self.ids = []
        print(">> Get all file's id")
        for bp_file in self.bpseq_file_list:
            _id = bp_file.split("/")[1]
            _id = _id.split(".")[0]
            self.ids.append(_id)
        
        for dp_file in self.dp_file_list:
            _id = dp_file.split("/")[1]
            _id = _id.split(".")[0]
            self.ids.append(_id)
            
        # eliminate redundant id
        self.ids = list(set(self.ids))
        print("number of ids : {}".format(len(self.ids)))
        X,L,C,A,N,S,E,ID = [],[],[],[],[],[],[],[]
        
        print(">> preprocess bpseq, dp format files")
        for i, _id in enumerate(self.ids):
            if i % 100 == 0:
                print("processed {} / {}".format(i, len(self.ids)))
                
            bp_file = self.bpseq_path + "/" + _id + "." + self.bpseq_type
            dp_file = self.dp_path    + "/" + _id + "." + self.dp_type
            
            bp_ret  = self.bpseq_encoding(bp_file)
            if len(bp_ret) == 2: 
                #print("error 1")
                continue 
            else: 
                (_, onehot_letters, letters, connections, adj, seq_length) = bp_ret
            
            dp_ret = self.dp_encoding(dp_file)
            if len(dp_ret) == 0 :
                #print("error 2")
                continue
            else: 
                (_, dot , onehot_dot) = dp_ret
            
            dot_elements = dotbracket_to_elements(dot)
            dot_elements = elements_structure_encoding(dot_elements)
            
            X.append(onehot_letters)
            L.append(letters)
            C.append(connections)
            A.append(adj)
            N.append(seq_length)
            S.append(onehot_dot)
            E.append(dot_elements)
            ID.append(_id)
        
        
        np.savez(self.npz, # npz file path
                 X=np.array(X), # X : RNA sequence(onehot encoded)
                 L=np.array(L), # L : RNA sequence letter (not encoded)
                 C=np.array(C), # C : RNA sequence's secondary structure link info (dictionary)
                 A=np.array(A), # A : RNA sequence's secondary structure link adjacency matrix (scipy.sparse.coo_matrix)
                 N=np.array(N), # N : RNA sequence's length 
                 S=np.array(S), # S : RNA sequence's secondary structure dot-bracket format (onehot encoded)
                 E=np.array(E), # E : RNA sequence's secondary structure dot-bracket base structure elements (onehot encoded)
                 ID = np.array(ID)) # ids
        
        print("Completely saved -> {}".format(self.npz))
        
    def bpseq_encoding(self, bpseq_file):
        # Split file path to get ID of bpseq-file
        data_id = bpseq_file.split("/")[1]
        data_id = data_id.split(".")[0]
        
        # Load file
        f = open(bpseq_file,"r")
        
        # Check the file error 
        try:
            f_lines = f.readlines()
        except Exception as ex :
            return [data_id, "Data read error"]
        
        # return dataset
        onehot_letters = []
        letters = []
        rows = []
        cols = []
        data = []
        connections = defaultdict(lambda: 0)
        
        for line in f_lines:
            if line[0] not in set(["#"," "]):
                idx, seq, link_idx = line.split()
                idx = int(idx)
                link_idx = int(link_idx)
                seq = seq.upper()
                
                # Check the invalid bpseq file
                if seq not in "ACGU":
                    return [data_id,"Not ACGU format"]
                
                letters.append(seq)
                onehot_letters.append(self.onehot_RNA[seq])
                
                connections[idx] = link_idx
                
                if link_idx > 0:
                    rows.append(idx-1)
                    cols.append(link_idx-1)
                    data.append(1)
                
        # convert defaultdict to dict(for save as npz type)
        connections = dict(connections)
        
        # make adjacency matrix
        N           = len(letters)
        rows        = np.array(rows)
        cols        = np.array(cols)
        adj         = sp.coo_matrix((data,(rows, cols)),shape=(N,N))
        
        return (data_id, onehot_letters, letters, connections, adj, N)
        
   
    def dp_encoding(self, dp_file):
        # Split file path to get ID of bpseq-file
        data_id = dp_file.split("/")[1]
        data_id = data_id.split(".")[0]
        
        f = open(dp_file,"r")
        
        try:
            f_lines = f.readlines()
        except Exception as ex :
            return []
                
        lines = ""
        for line in f_lines:
            if line[0] not in set(["#"," "]):
                lines += line
        
        split_lines = lines.split()
        seq_list = split_lines[:len(split_lines)//2]
        dot_list = split_lines[len(split_lines)//2:]
        
        dot = ""
        onehot_dot = []
        for d in dot_list:
            dot += d
            onehot_dot.append(self.onehot_DOT[d])
        
        onehot_dot = np.array(onehot_dot)
        
        return (data_id, dot, onehot_dot)