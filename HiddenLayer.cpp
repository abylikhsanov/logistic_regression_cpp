//
// Created by Abyl Ikhsanov on 14/01/2018.
//

#include "HiddenLayer.h"
#include <iostream>
#include <math.h>

HiddenLayer::HiddenLayer(int size, int in, int out, std::vector<std::vector<double>> const& w, std::vector<double> const& b){
    _N = size; // The size of
    _n_in = in;
    _n_out = out;
    //std::vector<std::vector<double>> weights(_n_in, std::vector<double>(_n_out,-1));
    _W = w;
    _b = b;
}

double HiddenLayer::output(std::vector<double> const &x) {

    // x here is the input from the prev layer
    std::vector<double> y(_n_out,0);
    for(int i = 0; i<_n_out; i++){
        y[i] = 0;
        for(int j = 0; j<_n_in; j++){
            y[i] += _W[i][j] * x[j];
        }
        y[i] += _b[i];
    }

}