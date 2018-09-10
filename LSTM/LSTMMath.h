/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LSTMMath.h
 * Author: stuarttruax
 *
 * Created on November 27, 2016, 9:45 AM
 */

#ifndef LSTMMATH_H
#define LSTMMATH_H
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

//supporting math 
long double pow_x ( long double x , unsigned i );
long double sigma(long double x);
long double sigmaPrime(long double x);
long double tanhLSTM(long double x);
long double tanhPrime(long double x);
long double* pointwiseMult(long double* x,long double* y, long double* z, int N);
long double* matrixVectorMult(long double** A,vector<long double> x,long double* y, int R, int C); 
long double* matrixVectorMult(long double** A,long double* x,long double* y, int R, int C); 
long double** transpose(long double** A,long double** T, int R, int C);
long double** outerProduct(vector<long double> &x,vector<long double> &y, long double** z, int R, int C);
long double** outerProduct(long double* x,long double* y, long double** z, int R, int C);
vector<long double> doubleArray2Vector(long double* toCopy, int length);



#endif /* LSTMMATH_H */

