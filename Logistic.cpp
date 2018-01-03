//
// Created by Abyl Ikhsanov on 31/12/2017.
//

#include "Logistic.h"
#include <vector>
#include <ostream>
#include <math.h>

Logistic::Logistic(int size, int n, int out): N(size), n_in(n), n_out(out){

    // init Weights and bias;
    W.assign(n_out,std::vector<double>(n_out,1));
    b.assign(n_out,0);
    dy.assign(n_out,0);

}

Logistic::~Logistic() {}

void Logistic::train(std::vector<int> x, std::vector<int> y, double lr){
    std::vector<double> p_y_given_x(n_out,0);
    for(int i = 0; i<n_out; i++){
        for(int j = 0; j<n_in; j++){
            p_y_given_x[i] += x[j] * W[i][j];
        }
        p_y_given_x[i] += b[i];
    }
    softmax(p_y_given_x);

    // Finding the error and using gradient descent to modify the weights:

    for(int i = 0; i<n_out; i++){
        dy[i] = y[i] - p_y_given_x[i];
        for(int j = 0; j<n_in; j++){
            W[i][j] += (lr * dy[i] * x[j]) / N;
        }
        b[i] += lr * dy[i]/N;
    }
}

void Logistic::softmax(std::vector<double>& x) {

    double maximum{0.0};
    double sum{0.0};

    for(int i = 0; i<n_out; i++) std::max(maximum, x[i]);

    for(int i = 0; i<n_out; i++){
        x[i] = exp(x[i] - maximum);
        sum += x[i];
    }

    for(int i = 0; i<n_out; i++) x[i] /= sum;
}

void Logistic::predict(std::vector<int> x, std::vector<double> y) {

    for(int i = 0; i<n_out; i++){
        y[i] = 0;
        for(int j = 0; j<n_in; j++){
            y[i] += W[i][j] * x[j];
        }
        y[i] += b[i];
    }
    std::cout<<x[1]<<" "<<y[1]<<std::endl;
    softmax(y);
    std::cout<<y[0]<<" "<<y[1]<<std::endl;

}

double Logistic::return_error(int i) {
    return dy[i];
}

