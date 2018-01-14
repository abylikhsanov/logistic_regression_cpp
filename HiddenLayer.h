//
// Created by Abyl Ikhsanov on 14/01/2018.
//

#ifndef LOGISTIC_REGRESSION_HIDDENLAYER_H
#define LOGISTIC_REGRESSION_HIDDENLAYER_H
#include <vector>

class HiddenLayer {
public:
    int _N;
    int _n_in;
    int _n_out;
    std::vector<std::vector<double>> _W;
    std::vector<double> _b;

    HiddenLayer(int, int, int,std::vector<std::vector<double>> const&, std::vector<double> const&);
    double output(std::vector<double> const&);

};


#endif //LOGISTIC_REGRESSION_HIDDENLAYER_H
