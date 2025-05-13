# cut pursuit: a working-set strategy to compute piecewise constant functions on graphs
C/C++ implementation of the L0-cut pursuit algorithms with Matlab and Python interfaces.

![illustration](https://user-images.githubusercontent.com/1902679/34037816-738cf4ba-e18a-11e7-9343-7c27209b27e6.png)

Cut pursuit is a graph-cut-based working-set strategy to minimize functions regularized by graph-structured regularizers.
For _G_ = (_V_, _E_, _w_) a graph with edges weighted by _w_, the problem writes:  

    min<sub>_x_ ∈ _Ω_<sup>_V_</sup></sub>    _f_(_x_) + 
    ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
    _φ_(_x_<sub>_u_</sub> - _x_<sub>_v_</sub>)  

where _Ω_ is the space in which lie the values associated with each node.  

We distinguish two different cases for _φ_, corresponding to different implementations:  
- _φ_: _t_ ↦ |_t_|: the convex case, the regularizer is the __graph total variation__.
Implemented for many different functionals _f_, such as quadratic, ℓ<sub>1</sub>-norm, box constraints, simplex constraints, linear, smoothed Kullback–Leibler.
See repository [CP_PFDR_graph_d1](https://github.com/1a7r0ch3/CP_PFDR_graph_d1), by Hugo Raguet. It is well-suited for regularization and inverse problems with a low total variation prior.
- _φ_: _t_ ↦ _δ_(_t_ ≠ 0) = 1 - _δ_<sub>0</sub>(t): the nonconvex case, the regularizer is the weight of the cut between the adjacent constant components. It is well-suited for segmentation/partitioning tasks. This repository corresponds to this problem.

 Current implementation supports the following fidelity functions:

- quadratic fidelity: _φ_: _x_ ↦ ∑<sub>_v_ in _V_</sub>||_x_<sub>_v_</sub> - _y_<sub>_v_</sub>||² with y an observed value associated with node _v_ (best for partitioning)
- linear fidelity: _φ_: _x_ ↦ - ∑<sub>_v_ in _V_</sub><_x_<sub>_v_</sub>, _y_<sub>_v_</sub>> with _y_<sub>_v_</sub> a weight associated with node _v_
- Kullback leibler fidelity _φ_: _x_ ↦ ∑<sub>_v_ in _V_</sub> KL(_x_<sub>_v_</sub>, _p_<sub>_v_</sub>) with _p_<sub>_v_</sub> a probability associated with node _v_. Only apply when _Ω_ is a simplex 

# Requirement

You need boost 1.58, or 1.65 if you want the python wrapper.

```conda install -c anaconda boost```

# Compilation

### C++
make sure that you use the following CPPFLAGS: 
```set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -fopenmp -O3 -Wall -std=c++11")```

add ```include<./cut-pursuit/include/API.h>``` and call any of the interface functions.

### MATLAB
To compile the MATLAB mex file type the following in MATLAB in the workspace containing the ```cut-pursuit``` folder:

```
mkdir ./cut-pursuit/bin
addpath('./cut-pursuit/bin/')
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" cut-pursuit/mex/L0_cut_pursuit.cpp ...
    -output cut-pursuit/bin/L0_cut_pursuit
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" cut-pursuit/mex/L0_cut_pursuit_segmentation.cpp ...
    -output cut-pursuit/bin/L0_cut_pursuit_segmentation
```

You can test the compilation with the following minimal example:

```
n_nodes = 100;
y = rand(3,n_nodes);
Eu = 0:(n_nodes-2);
Ev = 1:(n_nodes-1);
edge_weight = ones(numel(Eu),1);
node_weight = ones(n_nodes,1);
  
[solution, in_component, components] = L0_cut_pursuit_segmentation(single(y), uint32(Eu), uint32(Ev), single(.1)...
    , single(edge_weight), single(node_weight), 1, 2, 2);

subplot(3,1,1)
imagesc(repmat(y, [1 1 1]))
title('input data')
subplot(3,1,2)
imagesc(solution)
title('piecewise constant approximation')
subplot(3,1,3)
imagesc(in_component')
title('components')

```

### Python
Compile the library from the ```cut-pursuit``` folder
```
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include  -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
This creates ```build/src/libcp.so``` which can be imported in python. see ```test.py``` to test it out.

# References:
Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
L. Landrieu and G. Obozinski, SIAM Journal on Imaging Science 2017, Vol. 10, No. 4 : pp. 1724-1766
[[hal link]](https://hal.archives-ouvertes.fr/hal-01306779)

Cut-pursuit algorithm for nonsmooth functionals regularized by graph total variation, H. Raguet and L. Landrieu, in preparation. 

if using the L0-cut pursuit algorithm with \Omega other than R, one must also cite:

A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds. Loic Landrieu, Hugo Raguet , Bruno Vallet , Clément Mallet, Martin Weinmann
