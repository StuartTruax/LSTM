/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "LSTMMath.h"
//static const int order = 30; 

long double pow_x ( long double x , unsigned i )
{
       long double prod=1;
       if ( i == 0 )
          return 1;
       while ( i )
       {
             prod*=x;
             i--;
       }
       return prod;
}


long double sigma(long double a)
{
    return 1/(1+exp(-a));   
}

long double sigmaPrime(long double a)
{

    return (-exp(-a))/((1+exp(-a))*(1+exp(-a)));
   
}

long double tanhLSTM(long double a)
{

    return (exp(2*a)-1)/(exp(2*a)+1);  
}

long double tanhPrime(long double a)
{
    return 1 - pow_x(tanhLSTM(a),2);
        
}

long double* pointwiseMult(long double* a, long double* b, long double* c, int N)
{
    for(int i=0; i < N; i++)
    {
        c[i] = a[i]*b[i]; 
    }
    
    return c; 
}

long double* matrixVectorMult(long double** A, vector<long double> a, long double* b, int R, int C)
{
    long double rowSum;
    for(int i=0;i<R; i++)
    {
        rowSum = 0; 
        for(int j=0; j<C; j++)
        {
            rowSum = rowSum+A[i][j]*a[j];
        }
        b[i] = rowSum; 
    }
    
    return b; 
}
long double* matrixVectorMult(long double** A, long double* a, long double* b, int R, int C)
{
    long double rowSum;
    for(int i=0;i<R; i++)
    {
        rowSum = 0; 
        for(int j=0; j<C; j++)
        {
            rowSum = rowSum+A[i][j]*a[j];
        }
        b[i] = rowSum; 
    }
    
    return b; 
}

long double** outerProduct(vector<long double> &a, vector<long double> &b, long double** c, int R, int C)
{
    for(int i=0; i < R; i++)
    {
        for(int j=0; j < C; j++)
        {
            c[i][j] = a[i]*b[j]; 
        }
    }
    return c; 
}

long double** outerProduct(long double* a, long double* b, long double** c, int R, int C)
{
    for(int i=0; i < R; i++)
    {
        for(int j=0; j < C; j++)
        {
            c[i][j] = a[i]*b[j]; 
        }
    }
    return c; 
}

/* R is the number of rows in the transpose (columns in A)
 * C is the number of columns in the transpose (row in A)*/
long double** transpose(long double** A, long double** T, int R, int C)
{
   
    for(int i=0; i < R; i++)
    {
        for(int j=0; j < C; j++)
        {
            T[j][i] = A[i][j]; 
        }
    }
    
    return T; 
}

vector<long double> doubleArray2Vector(long double* toCopy, int length){
    
    vector<long double> toRet; 
    
    for(int i=0;i<length;i++)
    {
        toRet.push_back(toCopy[i]); 
    }
    
    return toRet;
}
