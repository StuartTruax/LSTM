/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LSTM.h
 * Author: stuarttruax
 * LSTM implementation based on notation and equations given in 
 * K. Greff, R. K. Srivastava, J. Koutnik, B.R. Steunebrink, J. Schmidhuber 
 * "LSTM: a search space odyssey", arXiv, 2015
 * Created on September 15, 2016, 10:57 AM
 */

#ifndef LSTM_H
#define LSTM_H
#include "LSTMMath.h"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

using namespace std; 

class LSTM{
public:
    LSTM(int N,int M);
   ~LSTM();
   
    void setInput(vector<double> input); 
    vector<double> getOutput(); 
    
    vector<vector<long double> > getZ_t();
    vector<vector<long double> > getI_t();
    vector<vector<long double> > getF_t();
    vector<vector<long double> > getC_t();
    vector<vector<long double> > getO_t();
    vector<vector<long double> > getY_t();
    vector<vector<long double> > getX_t();
    
    vector<vector<long double> > getW_z();
    vector<vector<long double> > getW_i();
    vector<vector<long double> > getW_f();
    vector<vector<long double> > getW_o();
    
    vector<vector<long double> > getR_z();
    vector<vector<long double> > getR_i();
    vector<vector<long double> > getR_f();
    vector<vector<long double> > getR_o();
    
    vector<long double> getB_z(); 
    vector<long double> getB_i(); 
    vector<long double> getB_f(); 
    vector<long double> getB_o(); 
    
    vector<long double> getP_i(); 
    vector<long double> getP_f(); 
    vector<long double> getP_o(); 
    
    
    void process();
    void setTrainingMode(bool isTraining);
    bool trainingStatus();
    
    void backPropagate(double eta,double momentum, vector<vector<double> > targetY); 
    void backPropagate_delta(double eta,double momentum, vector<vector<double> >& delta_in); 
    
    void serialize(ostream&);
    int  loadSerialization(istream&); 
    
    int getN();
    int getM(); 
    
private:
    //network dimensions
    int N; //number of LSTM blocks
    int M; //dimension of vector input (i.e. number of scalar inputs)
    
    
    
    //training mode boolean
    bool isTraining; 
    
    //time index during training interval; 
    int t_training;
    
    //Input Weights, NxM
    //matrices are of dimension NxM
    //N is the row dimension (highest level pointer)
    //M is the column dimension (2nd level pointer)
    long double** W_z,**W_i, **W_f, **W_o; 
    
    //Recurrent Weights, NxN
    //N is the row dimension (highest level pointer)
    //N is the column dimension (2nd level pointer)
    long double** R_z,**R_i, **R_f, **R_o;
    
    //Peephole weights, N
    long double* p_i, *p_f, *p_o;
    
    //bias weights, N
    long double* b_z, *b_i, *b_f, *b_o; 
    
    
   //all delta counterparts
   long double** dW_z, **dW_i, **dW_f, **dW_o; 
   long double** dR_z, **dR_i, **dR_f, **dR_o;
   
   long double* dp_i, *dp_f, *dp_o;
    
   long double* db_z, *db_i, *db_f, *db_o; 
   
   //previous versions for momentum weighting during update
   long double** dW_z_prev, **dW_i_prev, **dW_f_prev, **dW_o_prev; 
   long double** dR_z_prev, **dR_i_prev, **dR_f_prev, **dR_o_prev;
   
   long double* dp_i_prev, *dp_f_prev, *dp_o_prev;
    
   long double* db_z_prev, *db_i_prev, *db_f_prev, *db_o_prev; 
    
   
   ///training rate eta
   double eta; 
   //momentum
   double momentum; 
   
   //some typical values from the literature
   long double const p_i_val = 1.0; 
   long double const p_f_val = 1.0; 
   long double const p_o_val = 0.5; 
    
   long double const bias = 0.0; 
    
   //state vectors for training
   //state vectors are indexed from earliest timestep forward
   vector<vector<long double> > z_t; 
   vector<vector<long double> > i_t;
   vector<vector<long double> > f_t;
   vector<vector<long double> > c_t;
   vector<vector<long double> > o_t;
   vector<vector<long double> > y_t;
   vector<vector<long double> > x_t;
    
   //instantaneous z,i,f,o,c,y arrays 
   long double* z,*input,*f,*o,*c,*y,*x, *y_prev,*c_prev; 
    
   //delta vectors for backpropagation
   //delta vectors are indexed from latest timestep backward
   vector<vector<long double> > delta_t;
   vector<vector<long double> > dy_t;
   vector<vector<long double> > do_t;
   vector<vector<long double> > dc_t;
   vector<vector<long double> > df_t;
   vector<vector<long double> > di_t;
   vector<vector<long double> > dz_t;
    
   //instantaneous  dz,di,df,do,dc,dy, delta_t arrays
   long double* delta, *dz, *di, *df, *dc, *dO, *dy;
   
   void allocate(); 
   void initialize();
   
   //LSTM equations
   void zCalc();
   void iCalc();
   void fCalc(); 
   void cCalc(); 
   void oCalc(); 
   void yCalc(); 
   
   //backpropagation equations (in proper order)
   void dyCalc(int t);
   void doCalc(int t); 
   void dcCalc(int t);
   void dfCalc(int t); 
   void diCalc(int t);
   void dzCalc(int t); 
   
   //gradient calculations
   void dWCalc(int T);
   void dRCalc(int T);
   void dbCalc(int T);
   void dpiCalc(int T);
   void dpfCalc(int T);
   void dpoCalc(int T); 
   
   
   
};


#ifdef __cplusplus
extern "C" {
#endif




#ifdef __cplusplus
}
#endif

#endif /* LSTM_H */

