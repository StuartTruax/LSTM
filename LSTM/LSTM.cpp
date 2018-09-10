 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "LSTM.h"


LSTM::LSTM(int N, int M){
    this->N=N; 
    this->M=M; 
    isTraining = false; 
    allocate(); 
    initialize(); 
}

LSTM::~LSTM()
{
   //matrices are of dimension NxM
    //N is the row dimension (highest level pointer)
    //M is the column dimension (2nd level pointer)
    
      for(int i=0; i < N;i++)
    {
      free(R_z[i]); 
      free(R_i[i]);  
      free(R_f[i]);
      free(R_o[i]);   
      
      free(dR_z[i]); 
      free(dR_i[i]);
      free(dR_f[i]);
      free(dR_o[i]);
     
      free(dR_z_prev[i]);
      free(dR_i_prev[i]);
      free(dR_f_prev[i]);
      free(dR_o_prev[i]);
      
      free(W_z[i]);  
      free(W_i[i]); 
      free(W_f[i]);
      free(W_o[i]);
          
      free(dW_z[i]); 
      free(dW_i[i]);
      free(dW_f[i]);
      free(dW_o[i]);
        
      free(dW_z_prev[i]);
      free(dW_i_prev[i]);
      free(dW_f_prev[i]);
      free(dW_o_prev[i]);
    }
       
    //Weight matrices
    free(W_z);
    free(W_i);
    free(W_f);
    free(W_o);
    
    free(dW_z);
    free(dW_i);
    free(dW_f);
    free(dW_o);
    
   free(dW_z_prev);
   free(dW_i_prev);
   free(dW_f_prev);
   free(dW_o_prev);
   
   
   
    //Recurrent matrices
   free(R_z);
   free(R_i);
   free(R_f);
   free(R_o);
  
   free(dR_z);
   free(dR_i);
   free(dR_f);
   free(dR_o);
    
   free(dR_z_prev);
   free(dR_i_prev);
   free(dR_f_prev);
   free(dR_o_prev);
     
    //peephole weights
   
   free(p_i);
   free(p_f);
   free(p_o);
   
   free(dp_i);
   free(dp_f);
   free(dp_o);
   
   free(dp_i_prev);
   free(dp_f_prev);
   free(dp_o_prev);
     
    //bias weights
   
   free(b_z);
   free(b_i);
   free(b_f);
   free(b_o);
   
   free(db_z);
   free(db_i);
   free(db_f);
   free(db_o);
    
   free(db_z_prev);
   free(db_i_prev);
   free(db_f_prev);
   free(db_o_prev);

    ///instantaneous state arrays; 
   
   free(z);
   free(input);
   free(f);
   free(o);
   free(c);
   free(y);
   free(y_prev);
   free(c_prev);
   free(x);
    
    //instantaneous derivative arrays
   
   free(delta);
   free(dy);
   free(dz);
   free(di);
   free(dO);
   free(dc);
   free(df); 
}

int LSTM::getN()
{
    return N; 
}

int LSTM::getM()
{
    return M; 
}

void LSTM::allocate(){
    //matrices are of dimension NxM
    //N is the row dimension (highest level pointer)
    //M is the column dimension (2nd level pointer)
    
    //Weight matrices
    W_z = (long double**)malloc(sizeof(long double*)*N);
    W_i = (long double**)malloc(sizeof(long double*)*N);
    W_f = (long double**)malloc(sizeof(long double*)*N);
    W_o = (long double**)malloc(sizeof(long double*)*N);
    
    
    dW_z = (long double**)malloc(sizeof(long double*)*N);
    dW_i = (long double**)malloc(sizeof(long double*)*N);
    dW_f = (long double**)malloc(sizeof(long double*)*N);
    dW_o = (long double**)malloc(sizeof(long double*)*N);
    
    dW_z_prev = (long double**)malloc(sizeof(long double*)*N);
    dW_i_prev = (long double**)malloc(sizeof(long double*)*N);
    dW_f_prev = (long double**)malloc(sizeof(long double*)*N);
    dW_o_prev = (long double**)malloc(sizeof(long double*)*N);
    
    for(int i=0; i < N;i++)
    {
        W_z[i] = (long double*)malloc(sizeof(long double)*M);
        W_i[i] = (long double*)malloc(sizeof(long double)*M);
        W_f[i] = (long double*)malloc(sizeof(long double)*M);
        W_o[i] = (long double*)malloc(sizeof(long double)*M);
        
        dW_z[i] = (long double*)malloc(sizeof(long double)*M);
        dW_i[i] = (long double*)malloc(sizeof(long double)*M);
        dW_f[i] = (long double*)malloc(sizeof(long double)*M);
        dW_o[i] = (long double*)malloc(sizeof(long double)*M);
        
        dW_z_prev[i] = (long double*)malloc(sizeof(long double)*M);
        dW_i_prev[i] = (long double*)malloc(sizeof(long double)*M);
        dW_f_prev[i] = (long double*)malloc(sizeof(long double)*M);
        dW_o_prev[i] = (long double*)malloc(sizeof(long double)*M);
    }
    
    //Recurrent matrices
    R_z = (long double**)malloc(sizeof(long double*)*N); 
    R_i = (long double**)malloc(sizeof(long double*)*N); 
    R_f = (long double**)malloc(sizeof(long double*)*N); 
    R_o = (long double**)malloc(sizeof(long double*)*N);
    
    dR_z = (long double**)malloc(sizeof(long double*)*N); 
    dR_i = (long double**)malloc(sizeof(long double*)*N); 
    dR_f = (long double**)malloc(sizeof(long double*)*N); 
    dR_o = (long double**)malloc(sizeof(long double*)*N);
    
    dR_z_prev = (long double**)malloc(sizeof(long double*)*N); 
    dR_i_prev = (long double**)malloc(sizeof(long double*)*N); 
    dR_f_prev = (long double**)malloc(sizeof(long double*)*N); 
    dR_o_prev = (long double**)malloc(sizeof(long double*)*N);
    
    
      for(int i=0; i < N;i++)
    {
      R_z[i] = (long double*)malloc(sizeof(long double)*N); 
      R_i[i] = (long double*)malloc(sizeof(long double)*N); 
      R_f[i] = (long double*)malloc(sizeof(long double)*N); 
      R_o[i] = (long double*)malloc(sizeof(long double)*N);   
      
      
      dR_z[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_i[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_f[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_o[i] = (long double*)malloc(sizeof(long double)*N);
      
      dR_z_prev[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_i_prev[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_f_prev[i] = (long double*)malloc(sizeof(long double)*N); 
      dR_o_prev[i] = (long double*)malloc(sizeof(long double)*N);
    }
    
    
    //peephole weights
    p_i = (long double*)malloc(sizeof(long double)*N);
    p_f = (long double*)malloc(sizeof(long double)*N);
    p_o = (long double*)malloc(sizeof(long double)*N);
    
    dp_i = (long double*)malloc(sizeof(long double)*N);
    dp_f = (long double*)malloc(sizeof(long double)*N);
    dp_o = (long double*)malloc(sizeof(long double)*N);
    
    dp_i_prev = (long double*)malloc(sizeof(long double)*N);
    dp_f_prev = (long double*)malloc(sizeof(long double)*N);
    dp_o_prev = (long double*)malloc(sizeof(long double)*N);
    
    //bias weights
    b_z = (long double*)malloc(sizeof(long double)*N);
    b_i = (long double*)malloc(sizeof(long double)*N);
    b_f = (long double*)malloc(sizeof(long double)*N);
    b_o = (long double*)malloc(sizeof(long double)*N);
    
    
    db_z = (long double*)malloc(sizeof(long double)*N);
    db_i = (long double*)malloc(sizeof(long double)*N);
    db_f = (long double*)malloc(sizeof(long double)*N);
    db_o = (long double*)malloc(sizeof(long double)*N);
    
    db_z_prev = (long double*)malloc(sizeof(long double)*N);
    db_i_prev = (long double*)malloc(sizeof(long double)*N);
    db_f_prev = (long double*)malloc(sizeof(long double)*N);
    db_o_prev = (long double*)malloc(sizeof(long double)*N);
    
    ///instantaneous state arrays; 
    
    z = (long double*)malloc(sizeof(long double)*N);
    input = (long double*)malloc(sizeof(long double)*N);
    f = (long double*)malloc(sizeof(long double)*N);
    o = (long double*)malloc(sizeof(long double)*N);
    c = (long double*)malloc(sizeof(long double)*N);
    y = (long double*)malloc(sizeof(long double)*N);
    y_prev = (long double*)malloc(sizeof(long double)*N);
    c_prev = (long double*)malloc(sizeof(long double)*N);
    x = (long double*)malloc(sizeof(long double)*M);
    
    //instantaneous derivative arrays
    
    delta = (long double*)malloc(sizeof(long double)*N);
    dy = (long double*)malloc(sizeof(long double)*N);
    dz = (long double*)malloc(sizeof(long double)*N);
    di = (long double*)malloc(sizeof(long double)*N);
    dO = (long double*)malloc(sizeof(long double)*N);
    dc = (long double*)malloc(sizeof(long double)*N);
    df = (long double*)malloc(sizeof(long double)*N);
}

void LSTM::initialize()
{
    //Recurrent weight matrices R and input weight matrices W
    //set to the 1 matrix
    
    for(int i=0; i < N; i++ )
    {
        for(int j=0; j<M; j++)
        { 
             //W matrices
             W_z[i][j] = 1.0;
             W_i[i][j] = 1.0;
             W_f[i][j] = 1.0;
             W_o[i][j] = 1.0;
                      
             //dW matrices
             dW_z[i][j] = 0.0;
             dW_i[i][j] = 0.0;
             dW_f[i][j] = 0.0;
             dW_o[i][j] = 0.0;
             
             dW_z_prev[i][j] = 0.0;
             dW_i_prev[i][j] = 0.0;
             dW_f_prev[i][j] = 0.0;
             dW_o_prev[i][j] = 0.0;
             
        }
    }
    
    for(int i=0; i <N; i++)
    {
        b_z[i] = bias; 
        b_i[i] = bias; 
        b_f[i] = bias; 
        b_o[i] = bias; 
        
        p_i[i] = p_i_val; 
        p_f[i] = p_f_val; 
        p_o[i] = p_o_val; 
        
        
        db_z[i] = 0.0; 
        db_i[i] = 0.0; 
        db_f[i] = 0.0; 
        db_o[i] = 0.0; 
        
        db_z_prev[i] = 0.0; 
        db_i_prev[i] = 0.0; 
        db_f_prev[i] = 0.0; 
        db_o_prev[i] = 0.0; 
        
        dp_i[i] = 0.0; 
        dp_f[i] = 0.0; 
        dp_o[i] = 0.0; 
        
        dp_i_prev[i] = 0.0; 
        dp_f_prev[i] = 0.0; 
        dp_o_prev[i] = 0.0; 
        
        y_prev[i] = 0.0;
        c_prev[i] = 0.0;
        
        for(int j=0; j<N; j++)
        {
            //R matrices
            R_z[i][j] = 1.0; 
            R_i[i][j] = 1.0; 
            R_f[i][j] = 1.0; 
            R_o[i][j] = 1.0;
            
            //dR matrices
            dR_z[i][j] = 0.0; 
            dR_i[i][j] = 0.0; 
            dR_f[i][j] = 0.0; 
            dR_o[i][j] = 0.0; 
            
            dR_z_prev[i][j] = 0.0; 
            dR_i_prev[i][j] = 0.0; 
            dR_f_prev[i][j] = 0.0; 
            dR_o_prev[i][j] = 0.0; 
        }
        
        
    }
   
    
}

 ///Serialization format is:
// N,M
//followed by W_z, W_i, W_f, W_o
//followed by R_z, R_i, R_f, R_o
//followed by b_z, b_i, b_f, b_o
//followed by p_i, p_f, p_o
//for matrices, each line is a row with comma-separated column entries
//for vectors, each line is the vector with comma-separated entries
//stream abrogated with EOF

void LSTM::serialize(ostream& os){
   
    int i,j; 
    
    if(os.good())
    {
        os<<N<<","<<M<<"\n"; 
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<M; j++)
            {
                if(j==M-1)
                    os<<W_z[i][j]<<"\n";
                else 
                    os<<W_z[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<M; j++)
            {
                if(j==M-1)
                    os<<W_i[i][j]<<"\n";
                else 
                    os<<W_i[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<M; j++)
            {
                if(j==M-1)
                    os<<W_f[i][j]<<"\n";
                else 
                    os<<W_f[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<M; j++)
            {
                if(j==M-1)
                    os<<W_o[i][j]<<"\n";
                else 
                    os<<W_o[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<N; j++)
            {
                if(j==N-1)
                    os<<R_z[i][j]<<"\n";
                else 
                    os<<R_z[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
            for(j=0; j<N; j++)
            {
                if(j==N-1)
                    os<<R_i[i][j]<<"\n";
                else 
                    os<<R_i[i][j]<<",";
            }
        }
        
         for(i=0; i < N; i++)
        {
            for(j=0; j<N; j++)
            {
                if(j==N-1)
                    os<<R_f[i][j]<<"\n";
                else 
                    os<<R_f[i][j]<<",";
            }
        }
        
          for(i=0; i < N; i++)
        {
            for(j=0; j<N; j++)
            {
                if(j==N-1)
                    os<<R_o[i][j]<<"\n";
                else 
                    os<<R_o[i][j]<<",";
            }
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<b_z[i]<<"\n";
                else 
                    os<<b_z[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<b_i[i]<<"\n";
                else 
                    os<<b_i[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<b_f[i]<<"\n";
                else 
                    os<<b_f[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<b_o[i]<<"\n";
                else 
                    os<<b_o[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<p_i[i]<<"\n";
                else 
                    os<<p_i[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<p_f[i]<<"\n";
                else 
                    os<<p_f[i]<<",";
           
        }
        
        for(i=0; i < N; i++)
        {
                if(i==N-1)
                    os<<p_o[i]<<"\n";
                else 
                    os<<p_o[i]<<",";
           
        }
        
        os<<EOF; 
    }
    
}



bool LSTM::trainingStatus()
{
    return isTraining; 
}

void LSTM::setTrainingMode(bool isTraining)
{
    this->isTraining = isTraining; 
    
    //reset time index for training
    if(isTraining)
    {
        int i; 
        for(i=0; i < N; i++)
        {
            y[i] =0.0; 
            y_prev[i] = 0.0; 
            c[i] = 0.0;
            c_prev[i] = 0.0; 
            z[i] = 0.0;
            input[i] = 0.0;
            f[i] = 0.0;
            o[i] = 0.0; 
        }
        
        for(i=0; i < M; i++)
        {
            x[i] = 0.0; 
        }
        
        t_training = 0; 
    }
    ///free all state vectors; 
    else if((!isTraining))
    {
        z_t.clear();
        i_t.clear();
        f_t.clear();
        c_t.clear();
        o_t.clear();
        y_t.clear();
        
        int i; 
        for(i=0; i < N; i++)
        {
            y[i] =0.0; 
            y_prev[i] = 0.0; 
            c[i] = 0.0;
            c_prev[i] = 0.0; 
            z[i] = 0.0;
            input[i] = 0.0;
            f[i] = 0.0;
            o[i] = 0.0;
        }
        
        for(i=0; i < M; i++)
        {
            x[i] = 0.0; 
        }
        t_training=0;
        /*
        for(int i=0; i < N ; i++)
            for(int j=0; j < M; j++)
            {
                dW_z_prev[i][j] = 0.0;
                dW_i_prev[i][j] = 0.0;
                dW_f_prev[i][j] = 0.0;
                dW_o_prev[i][j] = 0.0; 
            }
        for(int i=0; i < N ; i++)
            for(int j=0; j < N; j++)
            {
                dR_z_prev[i][j] = 0.0;
                dR_i_prev[i][j] = 0.0;
                dR_f_prev[i][j] = 0.0;
                dR_o_prev[i][j] = 0.0; 
            }
    
        //clear bias and peephole deltas
        for(int i=0; i <N; i++)
        {
            db_z_prev[i] = 0.0; 
            db_i_prev[i] = 0.0; 
            db_f_prev[i] = 0.0; 
            db_o_prev[i] = 0.0; 
        
            dp_i_prev[i] = 0.0; 
            dp_f_prev[i] = 0.0; 
            dp_o_prev[i] = 0.0; 
        }
        */
    }
}


 void LSTM::setInput(vector<double> input)
 {
   for(int i=0; i < M; i++)
    {
        x[i] = input[i];
    }
 }


vector<double> LSTM::getOutput()
{
    vector<double> toRet(N); 
    
    for(int i=0; i < N; i++)
    {
        toRet[i] = y[i];
    }
    
    return toRet; 
    
}

vector<vector<long double> > LSTM::getZ_t(){
    
    return z_t; 
}

vector<vector<long double> > LSTM::getI_t(){
    
    return i_t; 
}

vector<vector<long double> > LSTM::getF_t(){
    
    return f_t; 
}

vector<vector<long double> > LSTM::getC_t(){
    
    return c_t; 
}

vector<vector<long double> > LSTM::getO_t(){
    
    return o_t; 
}



vector<vector<long double> > LSTM::getY_t(){
    
    return y_t; 
}

vector<vector<long double> > LSTM::getX_t(){
    
    return x_t; 
}


vector<vector<long double> > LSTM::getW_z(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(W_z[i],M)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getW_i(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(W_i[i],M)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getW_f(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(W_f[i],M)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getW_o(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(W_o[i],M)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getR_z(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(R_z[i],N)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getR_i(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(R_i[i],N)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getR_f(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(R_f[i],N)); 
    }
    
    return toRet; 
}

vector<vector<long double> > LSTM::getR_o(){
    vector<vector<long double> > toRet; 
    
    for(int i=0;i< N; i++)
    {
        toRet.push_back(doubleArray2Vector(R_o[i],N)); 
    }
    return toRet; 
}

vector<long double> LSTM::getB_z()
{
    
    return doubleArray2Vector(b_z,N);
}

vector<long double> LSTM::getB_i()
{
    
    return doubleArray2Vector(b_i,N);
}

vector<long double> LSTM::getB_f()
{
    
    return doubleArray2Vector(b_f,N);
}

vector<long double> LSTM::getB_o()
{
    
    return doubleArray2Vector(b_o,N);
}

vector<long double> LSTM::getP_i()
{
    
    return doubleArray2Vector(p_i,N);
}

vector<long double> LSTM::getP_f()
{
    
    return doubleArray2Vector(p_f,N);
}

vector<long double> LSTM::getP_o()
{
    
    return doubleArray2Vector(p_o,N);
}

void LSTM::process()
{ 
    //forward pass calculations in dependency-respecting order
    //calculate z_t
    zCalc(); 
    //calculate i_t
    iCalc(); 
    //calculate f_t
    fCalc(); 
    // calculate c_t
    cCalc(); 
    // calculate o_t
    oCalc(); 
    // calculate y_t
    yCalc(); 
    
    //copy y to y_prev 
    //and  c to c_prev
    for(int i=0; i < N ; i++)
    {
        y_prev[i]=y[i];
        c_prev[i] = c[i];        
    }
  
    
    //if isTraining, save z,i,f,c,o,y through a memcopy 
    // and push onto the respective arrays
    if(isTraining)
    {  
        z_t.push_back(doubleArray2Vector(z,N));
        i_t.push_back(doubleArray2Vector(input,N));
        f_t.push_back(doubleArray2Vector(f,N));
        c_t.push_back(doubleArray2Vector(c,N));
        o_t.push_back(doubleArray2Vector(o,N));
        y_t.push_back(doubleArray2Vector(y,N));
        x_t.push_back(doubleArray2Vector(x,M));
        
        t_training = t_training+1; 
        
        
    }
    
    
}

void LSTM::backPropagate(double eta, double momentum, vector<vector<double> > targetY){
    
    this->eta = eta;
    this->momentum = momentum; 
    ////////////////////////////////////////////////////////////
    ///calculate error vector
    //error vector will be indexed with latest time steps FIRST
    int i,j; 
    int stepsBack = 1; 
    int T = targetY.size(); 
    for(i=(T-1); i>=0; i-- )
    {
        for(j=0; j<N ; j++)
        {
            delta[j] = targetY[i][j]-y_t[t_training-stepsBack][j]; 
        }
        
        delta_t.push_back(doubleArray2Vector(delta,N)); 
        stepsBack++;  
    }  
    
    ////////////////////////////////////////////////////////////////
    ///calculate individual delta vectors for backpropagation
    for(i=0; i < T;i++)
    {    
        dyCalc(i);
        doCalc(i); 
        dcCalc(i);
        dfCalc(i); 
        diCalc(i);
        dzCalc(i);    
    }
    ///calculate the weight gradients
    dWCalc(T);
    dRCalc(T);
    dbCalc(T);
    dpiCalc(T);
    dpfCalc(T);
    dpoCalc(T);
    ////////////////////////////////////////////////////////////////
    ///add eta-scaled gradients to current weight vectors 
    
    ///Weight updates
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            W_z[i][j] += -momentum*dW_z_prev[i][j]-eta*dW_z[i][j];
            W_i[i][j] += -momentum*dW_i_prev[i][j]-eta*dW_i[i][j];
            W_f[i][j] += -momentum*dW_f_prev[i][j]-eta*dW_f[i][j];
            W_o[i][j] += -momentum*dW_o_prev[i][j]-eta*dW_o[i][j]; 
        }
    
     ///Recurrent updates
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            R_z[i][j] += -momentum*dR_z_prev[i][j]-eta*dR_z[i][j];
            R_i[i][j] += -momentum*dR_i_prev[i][j]-eta*dR_i[i][j];
            R_f[i][j] += -momentum*dR_f_prev[i][j]-eta*dR_f[i][j];
            R_o[i][j] += -momentum*dR_o_prev[i][j]-eta*dR_o[i][j]; 
        }
    
    //bias and peephole updates
    for(int i=0; i <N; i++)
    {
        b_z[i] += -momentum*db_z_prev[i]-eta*db_z[i]; 
        b_i[i] += -momentum*db_i_prev[i]-eta*db_i[i]; 
        b_f[i] += -momentum*db_f_prev[i]-eta*db_f[i]; 
        b_o[i] += -momentum*db_o_prev[i]-eta*db_o[i]; 
        
        p_i[i] += -momentum*dp_i_prev[i]-eta*dp_i[i]; 
        p_f[i] += -momentum*dp_f_prev[i]-eta*dp_f[i]; 
        p_o[i] += -momentum*dp_o_prev[i]-eta*dp_o[i]; 
    }
    ///////////////////////////////////////////////////////
    ///Copy the current updates to the corresponding _prev weights
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            dW_z_prev[i][j] =dW_z[i][j];
            dW_i_prev[i][j] =dW_i[i][j] ;
            dW_f_prev[i][j] =dW_f[i][j];
            dW_o_prev[i][j] =dW_o[i][j]; 
        }
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            dR_z_prev[i][j] = dR_z[i][j];
            dR_i_prev[i][j] = dR_i[i][j] ;
            dR_f_prev[i][j] = dR_f[i][j];
            dR_o_prev[i][j] = dR_o[i][j]; 
        }
    
     //clear bias and peephole deltas
    for(int i=0; i <N; i++)
    {
        db_z_prev[i] = db_z[i]; 
        db_i_prev[i] = db_i[i]; 
        db_f_prev[i] = db_f[i] ; 
        db_o_prev[i] = db_o[i]; 
        
        dp_i_prev[i] = dp_i[i]; 
        dp_f_prev[i] = dp_f[i]; 
        dp_o_prev[i] = dp_o[i]; 
    }
    
    ////////////////////////////////////////////////////////////
    ///free all delta vectors and clear weight gradients to 0
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            dW_z[i][j] = 0.0;
            dW_i[i][j] = 0.0;
            dW_f[i][j] = 0.0;
            dW_o[i][j] = 0.0; 
        }
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            dR_z[i][j] = 0.0;
            dR_i[i][j] = 0.0;
            dR_f[i][j] = 0.0;
            dR_o[i][j] = 0.0; 
        }
    
     //clear bias and peephole deltas
    for(int i=0; i <N; i++)
    {
        db_z[i] = 0.0; 
        db_i[i] = 0.0; 
        db_f[i] = 0.0; 
        db_o[i] = 0.0; 
        
        dp_i[i] = 0.0; 
        dp_f[i] = 0.0; 
        dp_o[i] = 0.0; 
    }
    
        
         
        for(int i=0; i<T;i++)
        {
             
             dz_t.pop_back();
             di_t.pop_back();
             df_t.pop_back();
             dc_t.pop_back();
             do_t.pop_back();
             dy_t.pop_back();
             
        }
    
    
}

///delta is assumed to have last timestep last
void LSTM::backPropagate_delta(double eta, double momentum, vector<vector<double> >& delta_in){
    
    this->eta = eta;
    this->momentum = momentum; 
    ////////////////////////////////////////////////////////////
    ///calculate error vector
    //error vector will be indexed with latest time steps FIRST
    int i,j; 
    int stepsBack = 1; 
    int T = delta_in.size(); 
    for(i=(T-1); i>=0; i-- )
    {
         for(j=0; j<N ; j++)
        {
            delta[j] = delta_in[i][j]; 
        }
         
        delta_t.push_back(doubleArray2Vector(delta,N)); 
        stepsBack++;  
    }  
    
    ////////////////////////////////////////////////////////////////
    ///calculate individual delta vectors for backpropagation
    for(i=0; i < T;i++)
    {    
        dyCalc(i);
        doCalc(i); 
        dcCalc(i);
        dfCalc(i); 
        diCalc(i);
        dzCalc(i);    
    }
    ///calculate the weight gradients
    dWCalc(T);
    dRCalc(T);
    dbCalc(T);
    dpiCalc(T);
    dpfCalc(T);
    dpoCalc(T);
    ////////////////////////////////////////////////////////////////
    ///add eta-scaled gradients to current weight vectors 
    
    ///Weight updates
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            W_z[i][j] += -momentum*dW_z_prev[i][j]-eta*dW_z[i][j];
            W_i[i][j] += -momentum*dW_i_prev[i][j]-eta*dW_i[i][j];
            W_f[i][j] += -momentum*dW_f_prev[i][j]-eta*dW_f[i][j];
            W_o[i][j] += -momentum*dW_o_prev[i][j]-eta*dW_o[i][j]; 
        }
    
     ///Recurrent updates
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            R_z[i][j] += -momentum*dR_z_prev[i][j]-eta*dR_z[i][j];
            R_i[i][j] += -momentum*dR_i_prev[i][j]-eta*dR_i[i][j];
            R_f[i][j] += -momentum*dR_f_prev[i][j]-eta*dR_f[i][j];
            R_o[i][j] += -momentum*dR_o_prev[i][j]-eta*dR_o[i][j]; 
        }
    
    //bias and peephole updates
    for(int i=0; i <N; i++)
    {
        b_z[i] += -momentum*db_z_prev[i]-eta*db_z[i]; 
        b_i[i] += -momentum*db_i_prev[i]-eta*db_i[i]; 
        b_f[i] += -momentum*db_f_prev[i]-eta*db_f[i]; 
        b_o[i] += -momentum*db_o_prev[i]-eta*db_o[i]; 
        
        p_i[i] += -momentum*dp_i_prev[i]-eta*dp_i[i]; 
        p_f[i] += -momentum*dp_f_prev[i]-eta*dp_f[i]; 
        p_o[i] += -momentum*dp_o_prev[i]-eta*dp_o[i]; 
    }
    ///////////////////////////////////////////////////////
    ///Copy the current updates to the corresponding _prev weights
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            dW_z_prev[i][j] =dW_z[i][j];
            dW_i_prev[i][j] =dW_i[i][j] ;
            dW_f_prev[i][j] =dW_f[i][j];
            dW_o_prev[i][j] =dW_o[i][j]; 
        }
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            dR_z_prev[i][j] = dR_z[i][j];
            dR_i_prev[i][j] = dR_i[i][j] ;
            dR_f_prev[i][j] = dR_f[i][j];
            dR_o_prev[i][j] = dR_o[i][j]; 
        }
    
     //clear bias and peephole deltas
    for(int i=0; i <N; i++)
    {
        db_z_prev[i] = db_z[i]; 
        db_i_prev[i] = db_i[i]; 
        db_f_prev[i] = db_f[i] ; 
        db_o_prev[i] = db_o[i]; 
        
        dp_i_prev[i] = dp_i[i]; 
        dp_f_prev[i] = dp_f[i]; 
        dp_o_prev[i] = dp_o[i]; 
    }
    
    ////////////////////////////////////////////////////////////
    ///free all delta vectors and clear weight gradients to 0
    for(int i=0; i < N ; i++)
        for(int j=0; j < M; j++)
        {
            dW_z[i][j] = 0.0;
            dW_i[i][j] = 0.0;
            dW_f[i][j] = 0.0;
            dW_o[i][j] = 0.0; 
        }
    for(int i=0; i < N ; i++)
        for(int j=0; j < N; j++)
        {
            dR_z[i][j] = 0.0;
            dR_i[i][j] = 0.0;
            dR_f[i][j] = 0.0;
            dR_o[i][j] = 0.0; 
        }
    
     //clear bias and peephole deltas
    for(int i=0; i <N; i++)
    {
        db_z[i] = 0.0; 
        db_i[i] = 0.0; 
        db_f[i] = 0.0; 
        db_o[i] = 0.0; 
        
        dp_i[i] = 0.0; 
        dp_f[i] = 0.0; 
        dp_o[i] = 0.0; 
    }
    
        
         
        for(int i=0; i<T;i++)
        {
             
             dz_t.pop_back();
             di_t.pop_back();
             df_t.pop_back();
             dc_t.pop_back();
             do_t.pop_back();
             dy_t.pop_back();
             
        }
    
    
}

//in all the d<>Calc functions, keep in mind the state vectors have a 
//different time-indexing from the delta vectors!! (on account that the latter
// are populated last-to-first)
// in state vector indexing, t_training-1-t is the "current" time step
// in delta vector indexing, t is the current time step
// in state vector indexing, t_training-t is the "t+1" time step
// in delta vector indexing, t-1 is the t+1 timestep

void LSTM::dyCalc(int t)
{
    if(t == 0)
    {
        for(int i=0; i < N;i++)
        {
            dy[i] = delta_t[0][i];
        } 
        dy_t.push_back(doubleArray2Vector(dy,N));
    }
    else
    {
        long double* term1 = (long double*)malloc(sizeof(long double)*N);
        long double* term2 = (long double*)malloc(sizeof(long double)*N);
        long double* term3 = (long double*)malloc(sizeof(long double)*N);
        long double* term4 = (long double*)malloc(sizeof(long double)*N);
        
        //create and multiply the transposed matrices
        long double** temp_transpose = (long double**)malloc(sizeof(long double*)*N); 
        for(int i=0; i<N; i++)
        {
            temp_transpose[i] = (long double*)malloc(sizeof(long double)*N);
        }
        
        temp_transpose = transpose(R_z,temp_transpose,N,N);
        matrixVectorMult(temp_transpose,dz_t[t-1],term1,N,N);
        
        temp_transpose = transpose(R_i,temp_transpose,N,N);
        matrixVectorMult(temp_transpose,di_t[t-1],term2,N,N);
        
        temp_transpose = transpose(R_f,temp_transpose,N,N);
        matrixVectorMult(temp_transpose,df_t[t-1],term3,N,N);
        
        temp_transpose = transpose(R_o,temp_transpose,N,N);
        matrixVectorMult(temp_transpose,do_t[t-1],term4,N,N);
        
        for(int i=0; i < N;i++)
        {
            dy[i] = delta_t[t][i] +term1[i] + term2[i]+term3[i]+term4[i];
        }
        
       
        dy_t.push_back(doubleArray2Vector(dy,N));
        
        
        //memory cleanup operations
         free(term1);
         free(term2); 
         free(term3);
         free(term4);
         
        
        for(int i=0; i<N; i++)
        {
            free(temp_transpose[i]);
        }
         free(temp_transpose);
         
    }
    
}
   void LSTM::doCalc(int t){
       
       for(int i=0; i < N;i++)
       {
           dO[i] = dy_t[t][i]*tanhLSTM(c_t[t_training-1-t][i])*sigmaPrime(o_t[t_training-1-t][i]);
       }
       
       
       do_t.push_back(doubleArray2Vector(dO,N));
   }
   
   void LSTM::dcCalc(int t){
       if(t==0)
       {
            for(int i=0; i < N;i++)
            {
                dc[i] = dy_t[t][i]*o_t[t_training-1-t][i]*tanhPrime(c_t[t_training-1-t][i])
                   +p_o[i]*do_t[t][i];
            }
       }
       else{
           for(int i=0; i < N;i++)
            {
                dc[i] = dy_t[t][i]*o_t[t_training-1-t][i]*tanhPrime(c_t[t_training-1-t][i])
                   +p_o[i]*do_t[t][i] + p_i[i]*di_t[t-1][i]+p_f[i]*df_t[t-1][i]
                        +dc_t[t-1][i]*f_t[t_training-t][i];
            }
       }
       
    
       dc_t.push_back(doubleArray2Vector(dc,N));
       
   }
   void LSTM::dfCalc(int t)
   {
       if(t_training-2-t >=0)
       {
            for(int i=0; i < N;i++)
            {
                df[i] = dc_t[t][i]*c_t[t_training-2-t][i]*sigmaPrime(f_t[t_training-1-t][i]);
            }
       }
       else{
           
           for(int i=0; i < N;i++)
            {
               // df[i] = dc_t[t][i]*sigmaPrime(f_t[t_training-1-t][i]);
                  df[i] = 0; 
            }
       }
       
     
      df_t.push_back(doubleArray2Vector(df,N));
   }
   
   void LSTM::diCalc(int t){
       
     for(int i=0; i < N;i++)
     {
                di[i] = dc_t[t][i]*z_t[t_training-1-t][i]*sigmaPrime(i_t[t_training-1-t][i]);
     }
       
     
     di_t.push_back(doubleArray2Vector(di,N));   
   }
   
   void LSTM::dzCalc(int t){
       
       for(int i=0; i < N;i++)
     {
                dz[i] = dc_t[t][i]*i_t[t_training-1-t][i]*tanhPrime(z_t[t_training-1-t][i]);
     }
       
    
     dz_t.push_back(doubleArray2Vector(dz,N));
       
   } 
   
   
void LSTM::dWCalc(int T){
    //make temporary matrix
    long double** temp_matrix = (long double**)malloc(sizeof(long double*)*N); 
    for(int i=0; i<N; i++)
    {
        temp_matrix[i] = (long double*)malloc(sizeof(long double)*M);
    }
    
    //make z-weight gradients
    for(int t=0; t< T; t++)
    {
        temp_matrix = outerProduct(dz_t[t],x_t[t_training-1-t], temp_matrix,N,M);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<M; j++)
            {
                dW_z[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
    
    //make i-weight gradients
    for(int t=0; t< T; t++)
    {
        temp_matrix = outerProduct(di_t[t],x_t[t_training-1-t], temp_matrix,N,M);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<M; j++)
            {
                dW_i[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
    
    //make f-weight gradients
    for(int t=0; t< T; t++)
    {
        temp_matrix = outerProduct(df_t[t],x_t[t_training-1-t], temp_matrix,N,M);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<M; j++)
            {
                dW_f[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
    
    
    //make o-weight gradients
    for(int t=0; t< T; t++)
    {
        temp_matrix = outerProduct(do_t[t],x_t[t_training-1-t], temp_matrix,N,M);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<M; j++)
            {
                dW_o[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
    
    for(int i=0; i<N; i++)
    {
            free(temp_matrix[i]);
    }
    free(temp_matrix);
}

void LSTM::dRCalc(int T)
{
    //make temporary matrix
    long double** temp_matrix = (long double**)malloc(sizeof(long double*)*N); 
    for(int i=0; i<N; i++)
    {
        temp_matrix[i] = (long double*)malloc(sizeof(long double)*N);
    }
    
    //make z-recurrent gradients
    for(int t=1;t< T; t++)
    {
        temp_matrix = outerProduct(dz_t[t-1],y_t[t_training-1-t], temp_matrix,N,N);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<N; j++)
            {
                dR_z[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
     //make i-recurrent gradients
    for(int t=1; t< T; t++)
    {
        temp_matrix = outerProduct(di_t[t-1],y_t[t_training-1-t], temp_matrix,N,N);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<N; j++)
            {
                dR_i[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
       //make f-recurrent gradients
    for(int t=1; t< T; t++)
    {
        temp_matrix = outerProduct(df_t[t-1],y_t[t_training-1-t], temp_matrix,N,N);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<N; j++)
            {
                dR_f[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
      //make o-recurrent gradients
    for(int t=1; t< T; t++)
    {
        temp_matrix = outerProduct(do_t[t-1],y_t[t_training-1-t], temp_matrix,N,N);
    
        for(int i=0;i<N;i++)
        {
            for(int j=0; j<N; j++)
            {
                dR_o[i][j] += temp_matrix[i][j]; 
            }
        } 
    }
    
    for(int i=0; i<N; i++)
    {
            free(temp_matrix[i]);
    }
    free(temp_matrix);
}

void LSTM::dbCalc(int T)
    {
    //bias for z 
    for(int t=0; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            db_z[i]+= dz_t[t][i]; 
        } 
    }
    
    //bias for i 
    for(int t=0; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            db_i[i]+= di_t[t][i]; 
        } 
    }
    
    //bias for f 
    for(int t=0; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            db_f[i]+= df_t[t][i]; 
        } 
    }
    
    //bias for o 
    for(int t=0; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            db_o[i]+= do_t[t][i]; 
        } 
    }
    
}

void LSTM::dpiCalc(int T){
    for(int t=1; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            dp_i[i]+= c_t[t_training-1-t][i]*di_t[t-1][i]; 
        } 
    }
    
}

void LSTM::dpfCalc(int T){
    for(int t=1; t< T; t++)
    {
        for(int i=0;i<N;i++)
        {
            dp_f[i]+= c_t[t_training-1-t][i]*df_t[t-1][i]; 
        } 
    }
    
}

void LSTM::dpoCalc(int T){
    for(int t=0; t<T; t++)
    {
        for(int i=0;i<N;i++)
        {
            dp_o[i]+= c_t[t_training-1-t][i]*do_t[t][i]; 
        } 
    }
}

void LSTM::zCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    long double* term2 = (long double*)malloc(sizeof(long double)*N);
    
    term1=matrixVectorMult(W_z,x,term1,N,M);
    term2=matrixVectorMult(R_z,y_prev,term2,N,N);
    
    for(int i=0; i < N; i++)
    {
        z[i] = tanhLSTM(term1[i]+term2[i] + b_z[i]); 
    }
    free(term1);
    free(term2); 
}

void LSTM::iCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    long double* term2 = (long double*)malloc(sizeof(long double)*N);
    long double* term3 = (long double*)malloc(sizeof(long double)*N);
    
    term1=matrixVectorMult(W_i,x,term1,N,M);
    term2=matrixVectorMult(R_i,y_prev,term2,N,N);
    term3=pointwiseMult(p_i,c_prev,term3, N);
    
    for(int j=0; j < N; j++)
    {
        input[j] = sigma(term1[j]+term2[j]+term3[j]+b_i[j]); 
    }
    free(term1);
    free(term2); 
    free(term3);
}

void LSTM::fCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    long double* term2 = (long double*)malloc(sizeof(long double)*N);
    long double* term3 = (long double*)malloc(sizeof(long double)*N);
    
    term1=matrixVectorMult(W_f,x,term1,N,M);
    term2=matrixVectorMult(R_f,y_prev,term2,N,N);
    term3=pointwiseMult(p_f,c_prev,term3, N);
    
    for(int i=0; i < N; i++)
    {
        f[i] = sigma(term1[i]+term2[i]+term3[i]+b_f[i]); 
    }
    free(term1);
    free(term2); 
    free(term3);
}

void LSTM::cCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    long double* term2 = (long double*)malloc(sizeof(long double)*N);
    
    
    term1=pointwiseMult(z,input,term1, N);
    term2=pointwiseMult(c_prev,f,term2, N);
    
    for(int i=0; i < N; i++)
    {
        c[i] = term1[i]+term2[i]; 
    }
    free(term1);
    free(term2); 
}

void LSTM::oCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    long double* term2 = (long double*)malloc(sizeof(long double)*N);
    long double* term3 = (long double*)malloc(sizeof(long double)*N);
    
    term1=matrixVectorMult(W_o,x,term1,N,M);
    term2=matrixVectorMult(R_o,y_prev,term2,N,N);
    term3=pointwiseMult(p_o,c,term3, N);
    
    for(int i=0; i < N; i++)
    {
        o[i] = sigma(term1[i]+term2[i]+term3[i]+b_o[i]); 
    }
    free(term1);
    free(term2); 
    free(term3);
}


void LSTM::yCalc()
{
    long double* term1 = (long double*)malloc(sizeof(long double)*N);
    for(int i=0; i < N; i++)
    {
        term1[i] = tanhLSTM(c[i]);   
    }
    
    this->y=pointwiseMult(term1,o,y, N);
   
    free(term1); 
}



