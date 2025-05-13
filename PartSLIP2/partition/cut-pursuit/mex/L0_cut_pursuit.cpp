#include <iostream>
#include <vector>
#include "mex.h"
//#include <opencv2/opencv.hpp>
#include "../include/API.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalzed minimal 
//partition problem
//
//Cut Pursuit: fast algorithms to learn piecewise constant functions on 
//general weighted graphs, Loic Landrieu and Guillaume Obozinski,2016.
//
//Produce a piecewise constant approximation of signal $y$ structured
//by the graph G=(V,e,mu,w) with mu the node weight and w the edgeweight:
//argmin \sum_{i \IN V}{mu_i * phi(x_I, y_I)} 
//+ \sum_{(i,j) \IN E}{w_{i,j} 1(x_I != x_J)}
//
//phi(X,Y) the fidelity function (3 are implemented)
//(x != y) the funciton equal to 1 if x!=y and 0 else
//
// LOIC LANDRIEU 2017
//
//=======================SYNTAX===================================================
//
//solution = CutPursuitMex(y, Eu, Ev, lambda = 1, edgeWeight = [1 ... 1]
//                 , nodeWeight = [1 ... 1], , mode = 1, speed = 1, verbose = false)
// float y : the observed signal, NxD
// int Eu, Ev: the origin and destination of each node, Ex1
// float edgeWeight: the edge weight, Ex1
// float nodeWeight: the node weight, Nx1
// float lambda : the regularization strength, 1x1
// float mode : the fidelity function
//      0 : linear (for simplex bound data)
//      1 : quadratic (default)
//   0<a<1: KL with a smoothing (for simplex bound data)
// float speed : parametrization impacting performance
//      0 : slow but precise
//      1 : recommended (default)
//      2 : fast but approximated (no backward step)
//      3 : ludicrous - for prototyping (no backward step)
// bool verbose : verbosity
//      0 : silent
//      1 : recommended (default)
//      2 : chatty

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int nNod = mxGetN(prhs[0]);
    const int nObs = mxGetM(prhs[0]);
    const int nEdg = mxGetNumberOfElements(prhs[1]);
    const uint32_t *Eu = (uint32_t*) mxGetData(prhs[1]);
    const uint32_t *Ev = (uint32_t*) mxGetData(prhs[2]);
    plhs[0] = mxDuplicateArray(prhs[0]);
    //plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    
    const float * y           = (float *) mxGetData(prhs[0]);
    const float lambda        = (float) mxGetScalar(prhs[3]);
    const float * edgeWeight  = (float *) mxGetData(prhs[4]);
    const float * nodeWeight  = (float *) mxGetData(prhs[5]);
    const float mode          = (float) mxGetScalar(prhs[6]);
    const float speed         = (float) mxGetScalar(prhs[7]);
    const float verbose       = (float) mxGetScalar(prhs[8]);    
    float * solution          = (float *) mxGetData(plhs[0]);

    CP::cut_pursuit<float>(nNod, nEdg, nObs, y
      , Eu, Ev, edgeWeight, nodeWeight, solution, lambda, mode, speed,verbose);
}
