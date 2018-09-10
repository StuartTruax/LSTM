//
//  LSTMTest.h
//  LSTMTest
//  A class that recreates the unit tests previously implemented using the CPPUnit framework.
//  Creates various LSTM networks with various dimensions of input and output vectors.
//  Inputs taken from the .csv files stored in the Debug/tests directory
//
//  Created by Stuart Truax on 9/10/18.
//  Copyright © 2018 Stuart Truax. All rights reserved.
//

#ifndef LSTMTest_h
#define LSTMTest_h

#include <iosfwd>
#include <sstream>
#include "LSTM.h"

using namespace std; 

class LSTMTest{
    public:
    //initialize and destroy NNs
    void setUp();
    void tearDown() ;
    
    //testing functions for initialized NNs
    void testMethod();
    void testIterations();
    void testIterations_N_N();
    void testIterations_N_M();
    void testIterations_N_1();
    void testIterations_1_M();
    void testSerialization();
    void testLSTMMath();
    
    
    //distance and error calculation functions
    double errorFunction(vector<vector<double> > target, vector<double> y);
    double errorFunction(vector<vector<double> > target, vector<vector<double> >  y);
    double euclideanDistance1D(vector<vector<double> > target, vector<double> y);
    double euclideanNorm(vector<double> x);
    void gradientVector(vector<double> prevError, vector<double> currError, vector<double>& toRet);
    void calcErrorVector(vector<vector<double> > target, vector<double> y, vector<double>& toRet);
    double MSE1D(vector<vector<double> > target, vector<double> output);
    double pointWiseMaxSE(vector<vector<double> > target, vector<double> output);
    void loadTestVector1D(char* vectorPath,vector<vector<double > >& target, vector<double>& input);
    void testMatchToOutput();
    
    //supporting display functions
    void printMatrix(long double** A, int N, int M);
    void printMatrix(vector<vector<long double> > A);
    void printVector(vector<long double> y, int N);
    void printVector(long double* y, int N);
    void loadTestVectorND(char* vectorPath,vector<vector<double > >& target, vector<vector<double > >& input, int N, int M);
    double euclideanDistanceND(vector<vector<double> > target, vector<vector<double> >  y);
    
    private:
    int lstm_N;
    int lstm_1;
    int lstm_M;
    LSTM* lSTM;
    LSTM* lSTM_N_N;
    LSTM* lSTM_N_M;
    LSTM* lSTM_N_1;
    LSTM* lSTM_1_M;
    
};


#endif /* LSTMTest_h */
