//
// Created by Abyl Ikhsanov on 31/12/2017.
//

#ifndef LOGISTIC_REGRESSION_LOGISTIC_H
#define LOGISTIC_REGRESSION_LOGISTIC_H

#include <iostream>
#include <vector>
#include "HiddenLayer.h"

class Logistic {

    int N;
    int n_in;
    int n_out;
    std::vector< std::vector<double> > W;
    std::vector<int> b; // bias;
    std::vector<double> dy;
    std::vector<HiddenLayer> layers;


public:
    Logistic(int, int, int);
    ~Logistic();
    void train(std::vector<int>, std::vector<int>, double);
    void softmax(std::vector<double>&);
    void predict(std::vector<int>, std::vector<double>);
    double return_error(int);
};


#endif //LOGISTIC_REGRESSION_LOGISTIC_H
