#include <iostream>

#include "Logistic.h"
#include <numeric>
double slope(const std::vector<double> & x, const std::vector<double> &y){
    std::cout<<x.size()<<" "<<y.size()<<std::endl;
    if(x.size() != y.size()) return 0.0;

    long size = x.size();
    double avgX = std::accumulate(x.begin(), x.end(), 0.0) / size;
    double avgY = std::accumulate(y.begin(), y.end(), 0.0) / size;

    double numerator = 0.0;
    double denom = 0.0;

    for(int i = 0; i<size; i++){
        numerator += (x[i] - avgX)*(y[i] * avgY);
        denom += (x[i] -avgX)*(y[i] - avgX);
    }

    return numerator/denom;
}

template<class K>
void start(double lrate, int n_epochs, K train_n, K test_n, std::vector< std::vector<K> > train_x,
           std::vector< std::vector<K> > train_y){

    Logistic l(train_n, train_n, test_n);
    for(int i = 0; i<n_epochs; i++){
        std::cout<<"EPOCH "<<i;
        for(int j = 0; j<train_n; j++){
            l.train(train_x[j], train_y[j], lrate);
        }
        std::cout<<" accuracy:"<<l.return_error(i)<<std::endl;
    }
    l.predict({1,1,1,0,0,0},{0,0});
}
int main() {
    std::vector< std::vector<int> > train_x{{1, 1, 1, 0, 0, 0},
                                            {1, 0, 1, 0, 0, 0},
                                            {1, 1, 1, 0, 0, 0},
                                            {0, 0, 1, 1, 1, 0},
                                            {0, 0, 1, 1, 0, 0},
                                            {0, 0, 1, 1, 1, 0}};
    std::vector< std::vector<int> > y_train{{1, 0},
                                          {1, 0},
                                          {1, 0},
                                          {0, 1},
                                          {0, 1},
                                          {0, 1}};
    start(0.1,500,6,2,train_x,y_train);

    return 0;
}