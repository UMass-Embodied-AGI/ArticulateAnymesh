import sys
sys.path.append("../build/src")
import libcp
import numpy as np
obs = np.array([[10.,20.],[0.,1.],[0.,0. ]], dtype = 'float32')
source = np.array([0, 1], dtype = 'uint32')
target = np.array([1, 2], dtype = 'uint32')
edge_weight = np.array([[1., 1.]], dtype = 'float32')
[comp, in_com] = libcp.cutpursuit(obs, source, target, edge_weight, 1)

#hierarchical partition

obs = np.array([[0],[1],[5],[6],[20]], dtype='float32')
source = np.array([0,0,0,0,1,1,1,2,2,3], dtype='uint32')
target = np.array([1,2,3,4,2,3,4,3,4,4], dtype='uint32')
edge_weight = np.array([1,1,1,1,1,1,1,1,1,1], dtype='float32')
reg_str = np.array([1,5], dtype='float32')
cut_off = np.array([1,1], dtype='uint32')

hier = libcp.cutpursuit_hierarchy(x, Eu, Ev,  Ew ,rs, co, 0, 0.7)