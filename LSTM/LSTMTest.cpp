//
//  LSTMTest.cpp
//  LSTM
//
//  Created by Stuart Truax on 9/10/18.
//  Copyright © 2018 Stuart Truax. All rights reserved.
//

#include "LSTMTest.h"

void LSTMTest::setUp() {
    lstm_N = 3;
    lstm_M = 4;
    
    lSTM = new LSTM(1,1);
    lSTM_N_N = new LSTM(lstm_N,lstm_N);
    lSTM_N_M = new LSTM(lstm_N,lstm_M);
    lSTM_N_1 = new LSTM(lstm_N,1);
    lSTM_1_M = new LSTM(1,lstm_M);
    
    //lSTM_modY = new LSTM_modY(1,1);
    
    //clear test output directories
    FILE* fp = popen("rm tests/output/*","w");
    int status = pclose(fp);
}


void LSTMTest::tearDown() {
    delete lSTM;
    delete lSTM_N_N;
    delete lSTM_N_M;
    delete lSTM_N_1;
    delete lSTM_1_M;
}

void LSTMTest::testMethod() {
    char vectorPath[] = "tests/1_1_5_pulse_gauss.csv";
    
    vector<double> output;
    vector<double> input;
    vector<vector<double > > target;
    
    int i=0;
    int j=0;
    double eta = 0.01;
    double momentum = 0.5;
    
    loadTestVector1D(vectorPath,target, input);
    
    
    //
    ///////////////////////////////////////////
    //
    cout<<"Pass 0 \n";
    cout<<"Output from network: \n";
    
    lSTM->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp.push_back(input[i]);
        lSTM->setInput(temp);
        temp.clear();
        lSTM->process();
        output.push_back((lSTM->getOutput())[0]);
        cout<<output[i]<<"\n";
    }
    
    output.clear();
    
    ////////////////////////////////////////////
    cout<<"Back Propagation \n";
    lSTM->backPropagate(eta,momentum,target);
    lSTM->setTrainingMode(false);
    lSTM->setTrainingMode(true);
    
    
    /////////////////////////////////////////////
    cout<<"Pass 1 \n";
    cout<<"Output from network: \n";
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp.push_back(input[i]);
        lSTM->setInput(temp);
        temp.clear();
        lSTM->process();
        output.push_back((lSTM->getOutput())[0]);
        cout<<output[i]<<"\n";
    }
    output.clear();
    
    //////////////////////////////////////////////////
    cout<<"Back Propagation \n";
    lSTM->backPropagate(eta,momentum,target);
    lSTM->setTrainingMode(false);
    
    
    ////////////////////////////////////////////////
    cout<<"Pass 2 \n";
    cout<<"Output from network: \n";
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp.push_back(input[i]);
        lSTM->setInput(temp);
        temp.clear();
        lSTM->process();
        output.push_back((lSTM->getOutput())[0]);
        cout<<output[i]<<"\n";
    }
    output.clear();
    
}

void LSTMTest::testIterations()
{
    char vectorPath[] = "tests/1_1_5_pulse_gauss.csv";
    char outputPath[] = "tests/output/";
    char filename_buf[100];
    
    
    vector<double> output;
    vector<double> input;
    vector<vector<double > > target;
    
    vector<double> currError;
    vector<double> prevError;
    vector<double> errorGrad;
    
    int i=0;
    int j=0;
    
    int maxIterations = 1000;
    double eta = 0.01;
    double momentum = 0.05;
    double gradNormThreshold = 0.00001;
    double gradNorm = 0.0;
    
    double currSE = 0;
    double prevSE = 0;
    double minSE;
    double SEThreshold = 0.01;
    
    loadTestVector1D(vectorPath,target, input);
    //////////////////////////////////
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    for(i=0; i < target.size();i++)
    {
        currError.push_back(0.0);
        prevError.push_back(0.0);
        errorGrad.push_back(0.0);
    }
    
    
    lSTM->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp.push_back(input[i]);
        lSTM->setInput(temp);
        temp.clear();
        lSTM->process();
        output.push_back((lSTM->getOutput())[0]);
        cout<<"Y:"<<output[i]<<"\n";
    }
    
    
    
    for(j=0; j < maxIterations; j++)
    {
        output.clear();
        lSTM->backPropagate(eta,momentum,target);
        lSTM->setTrainingMode(false);
        lSTM->setTrainingMode(true);
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp.push_back(input[i]);
            lSTM->setInput(temp);
            temp.clear();
            lSTM->process();
            output.push_back((lSTM->getOutput())[0]);
        }
        
        
        ////////////////////////////////////
        ////Stopping Criteria 1 (norm of gradient on error surface)
        calcErrorVector(target,output,currError);
        gradientVector(prevError,currError,errorGrad);
        
        gradNorm = euclideanNorm(errorGrad);
        
        for(i=0; i < target.size();i++)
        {
            prevError[i] = currError[i];
        }
        
        //if(gradNorm < gradNormThreshold)
        //   break;
        
        ////////////////////////////////////
        ////Stopping Criteria 2 (change in MSE between epochs)
        
        currSE = errorFunction(target,output)/target.size();
        
        // if((abs(currSE-prevSE)/currSE) < SEThreshold)
        //     break;
        
        ////////////////////////////////////////////
        ////Stopping Criteria 3 (minimum in Euclidean distance)
        ////  thus far seems to work best
        if(j==0)
        {
            minSE = currSE;
        }
        else if(minSE < currSE)
        {
            break;
        }
        else if(currSE < minSE)
        {
            minSE = currSE;
            
        }
        
        cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
        /////////////////////////////////////////
        //Serialize each output instance
        //sprintf(filename_buf,"%sLSTM_cpp_%d.csv",outputPath,j);
        sprintf(filename_buf,"%s%d.csv",outputPath,j);
        //cout<<filename_buf;
        filebuf fb;
        fb.open(filename_buf,ios::out|ios::app);
        ostream os(&fb);
        lSTM->serialize(os);
        fb.close();
    }
    //////////////////////////////////////////////
    lSTM->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    cout<<" t | Y | Y_target | error_t \n";
    
    for(i=0; i < target.size();i++)
    {
        cout<<i<<"|"<<output[i]<<"|"<<target[i][0]<<"|"<<(target[i][0]-output[i])/target[i][0]<<"\n";
    }
    
}


void LSTMTest::testIterations_N_N()
{
    char vectorPath[] = "tests/3_3_5_pulse_gauss.csv";
    char outputPath[] = "tests/output/";
    char filename_buf[100];
    
    
    vector<vector<double > > output;
    vector<vector<double > > input;
    vector<vector<double > > target;
    
    vector<double> currError;
    vector<double> prevError;
    vector<double> errorGrad;
    
    int i=0;
    int j=0;
    int k=0;
    
    int maxIterations = 2000;
    double eta = 0.01;
    double momentum = 0.05;
    double gradNormThreshold = 0.00001;
    double gradNorm = 0.0;
    
    double currSE = 0;
    double prevSE = 0;
    double minSE;
    double SEThreshold = 0.01;
    
    loadTestVectorND(vectorPath,target, input, lstm_N,lstm_N);
    //////////////////////////////////
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    for(i=0; i < target.size();i++)
    {
        currError.push_back(0.0);
        prevError.push_back(0.0);
        errorGrad.push_back(0.0);
    }
    
    
    lSTM_N_N->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp = input[i];
        lSTM_N_N->setInput(temp);
        lSTM_N_N->process();
        output.push_back((lSTM_N_N->getOutput()));
        
        cout<<"Y:\n";
        for(k=0; k<lstm_N; k++)
        {
            cout<<output[i][k]<<",";
        }
        cout<<"\n";
    }
    
    
    
    for(j=0; j < maxIterations; j++)
    {
        output.clear();
        lSTM_N_N->backPropagate(eta,momentum,target);
        lSTM_N_N->setTrainingMode(false);
        lSTM_N_N->setTrainingMode(true);
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp = input[i];
            lSTM_N_N->setInput(temp);
            lSTM_N_N->process();
            output.push_back((lSTM_N_N->getOutput()));
        }
        
        
        ////////////////////////////////////
        ////Stopping Criteria 1 (norm of gradient on error surface)
        
        
        // calcErrorVector(target,output,currError,lstm_N);
        // gradientVector(prevError,currError,errorGrad,lstm_N);
        
        // gradNorm = euclideanNorm(errorGrad);
        
        // for(i=0; i < target.size();i++)
        // {
        //      prevError[i] = currError[i];
        // }
        
        //if(gradNorm < gradNormThreshold)
        //   break;
        
        ////////////////////////////////////
        ////Stopping Criteria 2 (change in MSE between epochs)
        
        // currSE = errorFunction(target,output)/target.size();
        
        // if((abs(currSE-prevSE)/currSE) < SEThreshold)
        //     break;
        
        ////////////////////////////////////////////
        ////Stopping Criteria 3 (minimum in Euclidean distance)
        ////  thus far seems to work best
        
        currSE = errorFunction(target,output);
        
        if(j==0)
        {
            minSE = currSE;
        }
        else if(minSE < currSE)
        {
            break;
        }
        else if(currSE < minSE)
        {
            minSE = currSE;
            
        }
        
        cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
        /////////////////////////////////////////
        //Serialize each output instance
        //sprintf(filename_buf,"%sLSTM_cpp_%d.csv",outputPath,j);
        sprintf(filename_buf,"%s%d.csv",outputPath,j);
        //cout<<filename_buf;
        filebuf fb;
        fb.open(filename_buf,ios::out|ios::app);
        ostream os(&fb);
        lSTM_N_N->serialize(os);
        fb.close();
    }
    //////////////////////////////////////////////
    lSTM_N_N->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    
    
    for(i=0; i < target.size();i++)
    {
        cout<<"t = "<<i<<":\n";
        cout<<" Y:\n";
        cout<<" ";
        for(j=0; j < lstm_N;j++)
        {
            cout<<output[i][j]<<", ";
        }
        cout<<"\n";
        cout<<" Y_target: \n";
        cout<<" ";
        for(j=0; j <lstm_N;j++)
        {
            cout<<target[i][j]<<", ";
        }
        cout<<"\n";
        
    }
    
}


void LSTMTest::testIterations_N_M()
{
    char vectorPath[] = "tests/3_4_5_pulse_gauss.csv";
    char outputPath[] = "tests/output/";
    char filename_buf[100];
    
    
    vector<vector<double > > output;
    vector<vector<double > > input;
    vector<vector<double > > target;
    
    vector<double> currError;
    vector<double> prevError;
    vector<double> errorGrad;
    
    int i=0;
    int j=0;
    int k=0;
    
    int maxIterations = 2000;
    double eta = 0.01;
    double momentum = 0.05;
    double gradNormThreshold = 0.00001;
    double gradNorm = 0.0;
    
    double currSE = 0;
    double prevSE = 0;
    double minSE;
    double SEThreshold = 0.01;
    
    loadTestVectorND(vectorPath,target, input, lstm_N,lstm_M);
    //////////////////////////////////
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    for(i=0; i < target.size();i++)
    {
        currError.push_back(0.0);
        prevError.push_back(0.0);
        errorGrad.push_back(0.0);
    }
    
    
    lSTM_N_M->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp = input[i];
        lSTM_N_M->setInput(temp);
        lSTM_N_M->process();
        output.push_back((lSTM_N_M->getOutput()));
        
        cout<<"Y:\n";
        for(k=0; k<lstm_N; k++)
        {
            cout<<output[i][k]<<",";
        }
        cout<<"\n";
    }
    
    
    
    for(j=0; j < maxIterations; j++)
    {
        output.clear();
        lSTM_N_M->backPropagate(eta,momentum,target);
        lSTM_N_M->setTrainingMode(false);
        lSTM_N_M->setTrainingMode(true);
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp = input[i];
            lSTM_N_M->setInput(temp);
            lSTM_N_M->process();
            output.push_back((lSTM_N_M->getOutput()));
        }
        
        
        ////////////////////////////////////
        ////Stopping Criteria 1 (norm of gradient on error surface)
        
        
        // calcErrorVector(target,output,currError,lstm_N);
        // gradientVector(prevError,currError,errorGrad,lstm_N);
        
        // gradNorm = euclideanNorm(errorGrad);
        
        // for(i=0; i < target.size();i++)
        // {
        //      prevError[i] = currError[i];
        // }
        
        //if(gradNorm < gradNormThreshold)
        //   break;
        
        ////////////////////////////////////
        ////Stopping Criteria 2 (change in MSE between epochs)
        
        // currSE = errorFunction(target,output)/target.size();
        
        // if((abs(currSE-prevSE)/currSE) < SEThreshold)
        //     break;
        
        ////////////////////////////////////////////
        ////Stopping Criteria 3 (minimum in Euclidean distance)
        ////  thus far seems to work best
        
        currSE = errorFunction(target,output);
        
        if(j==0)
        {
            minSE = currSE;
        }
        else if(minSE < currSE)
        {
            break;
        }
        else if(currSE < minSE)
        {
            minSE = currSE;
            
        }
        
        cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
        /////////////////////////////////////////
        //Serialize each output instance
        //sprintf(filename_buf,"%sLSTM_cpp_%d.csv",outputPath,j);
        sprintf(filename_buf,"%s%d.csv",outputPath,j);
        //cout<<filename_buf;
        filebuf fb;
        fb.open(filename_buf,ios::out|ios::app);
        ostream os(&fb);
        lSTM_N_M->serialize(os);
        fb.close();
    }
    //////////////////////////////////////////////
    lSTM_N_M->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    
    
    for(i=0; i < target.size();i++)
    {
        cout<<"t = "<<i<<":\n";
        cout<<" Y:\n";
        cout<<" ";
        for(j=0; j < lstm_N;j++)
        {
            cout<<output[i][j]<<", ";
        }
        cout<<"\n";
        cout<<" Y_target: \n";
        cout<<" ";
        for(j=0; j <lstm_N;j++)
        {
            cout<<target[i][j]<<", ";
        }
        cout<<"\n";
        
    }
    
    
    
    
    

}

void LSTMTest::testIterations_N_1()
{
    char vectorPath[] = "tests/1_3_5_pulse_gauss.csv";
    char outputPath[] = "tests/output/";
    char filename_buf[100];
    
    
    vector<vector<double > > output;
    vector<vector<double > > input;
    vector<vector<double > > target;
    
    vector<double> currError;
    vector<double> prevError;
    vector<double> errorGrad;
    
    int i=0;
    int j=0;
    int k=0;
    
    int maxIterations = 2000;
    double eta = 0.01;
    double momentum = 0.05;
    double gradNormThreshold = 0.00001;
    double gradNorm = 0.0;
    
    double currSE = 0;
    double prevSE = 0;
    double minSE;
    double SEThreshold = 0.01;
    
    loadTestVectorND(vectorPath,target, input, lstm_N,1);
    //////////////////////////////////
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    for(i=0; i < target.size();i++)
    {
        currError.push_back(0.0);
        prevError.push_back(0.0);
        errorGrad.push_back(0.0);
    }
    
    
    lSTM_N_1->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp = input[i];
        lSTM_N_1->setInput(temp);
        lSTM_N_1->process();
        output.push_back((lSTM_N_1->getOutput()));
        
        cout<<"Y:\n";
        for(k=0; k<lstm_N; k++)
        {
            cout<<output[i][k]<<",";
        }
        cout<<"\n";
    }
    
    
    
    for(j=0; j < maxIterations; j++)
    {
        output.clear();
        lSTM_N_1->backPropagate(eta,momentum,target);
        lSTM_N_1->setTrainingMode(false);
        lSTM_N_1->setTrainingMode(true);
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp = input[i];
            lSTM_N_1->setInput(temp);
            lSTM_N_1->process();
            output.push_back((lSTM_N_1->getOutput()));
        }
        
        
        ////////////////////////////////////
        ////Stopping Criteria 1 (norm of gradient on error surface)
        
        
        // calcErrorVector(target,output,currError,lstm_N);
        // gradientVector(prevError,currError,errorGrad,lstm_N);
        
        // gradNorm = euclideanNorm(errorGrad);
        
        // for(i=0; i < target.size();i++)
        // {
        //      prevError[i] = currError[i];
        // }
        
        //if(gradNorm < gradNormThreshold)
        //   break;
        
        ////////////////////////////////////
        ////Stopping Criteria 2 (change in MSE between epochs)
        
        // currSE = errorFunction(target,output)/target.size();
        
        // if((abs(currSE-prevSE)/currSE) < SEThreshold)
        //     break;
        
        ////////////////////////////////////////////
        ////Stopping Criteria 3 (minimum in Euclidean distance)
        ////  thus far seems to work best
        
        currSE = errorFunction(target,output);
        
        if(j==0)
        {
            minSE = currSE;
        }
        else if(minSE < currSE)
        {
            break;
        }
        else if(currSE < minSE)
        {
            minSE = currSE;
            
        }
        
        cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
        /////////////////////////////////////////
        //Serialize each output instance
        //sprintf(filename_buf,"%sLSTM_cpp_%d.csv",outputPath,j);
        sprintf(filename_buf,"%s%d.csv",outputPath,j);
        //cout<<filename_buf;
        filebuf fb;
        fb.open(filename_buf,ios::out|ios::app);
        ostream os(&fb);
        lSTM_N_1->serialize(os);
        fb.close();
    }
    //////////////////////////////////////////////
    lSTM_N_1->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    
    
    for(i=0; i < target.size();i++)
    {
        cout<<"t = "<<i<<":\n";
        cout<<" Y:\n";
        cout<<" ";
        for(j=0; j < lstm_N;j++)
        {
            cout<<output[i][j]<<", ";
        }
        cout<<"\n";
        cout<<" Y_target: \n";
        cout<<" ";
        for(j=0; j <lstm_N;j++)
        {
            cout<<target[i][j]<<", ";
        }
        cout<<"\n";
        
    }
    
    
    
    
    
}

void LSTMTest::testIterations_1_M()
{
    char vectorPath[] = "tests/4_1_5_pulse_gauss.csv";
    char outputPath[] = "tests/output/";
    char filename_buf[100];
    
    
    vector<vector<double > > output;
    vector<vector<double > > input;
    vector<vector<long double> > tempVec;
    vector<vector<double > > target;
    
    vector<double> currError;
    vector<double> prevError;
    vector<double> errorGrad;
    
    int i=0;
    int j=0;
    int k=0;
    
    int maxIterations = 1000;
    double eta = 0.01;
    double momentum = 0.05;
    double gradNormThreshold = 0.00001;
    double gradNorm = 0.0;
    
    double currSE = 0;
    double prevSE = 0;
    double minSE;
    double SEThreshold = 0.01;
    
    loadTestVectorND(vectorPath,target, input, 1,lstm_M);
    //////////////////////////////////
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    for(i=0; i < target.size();i++)
    {
        currError.push_back(0.0);
        prevError.push_back(0.0);
        errorGrad.push_back(0.0);
    }
    
    
    lSTM_1_M->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp = input[i];
        lSTM_1_M->setInput(temp);
        lSTM_1_M->process();
        output.push_back((lSTM_1_M->getOutput()));
        
        cout<<"Y:\n";
        for(k=0; k<1; k++)
        {
            cout<<output[i][k]<<",";
        }
        cout<<"\n";
    }
    
    
    
    for(j=0; j < maxIterations; j++)
    {
        
        output.clear();
        lSTM_1_M->backPropagate(eta,momentum,target);
        lSTM_1_M->setTrainingMode(false);
        lSTM_1_M->setTrainingMode(true);
        
        
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp = input[i];
            //cout<<temp[0]<<", "<<temp[1]<<", "<<temp[2]<<", "<<temp[3]<<"\n";
            lSTM_1_M->setInput(temp);
            lSTM_1_M->process();
            output.push_back((lSTM_1_M->getOutput()));
        }
        
        
        ////////////////////////////////////
        ////Stopping Criteria 1 (norm of gradient on error surface)
        
        
        // calcErrorVector(target,output,currError,lstm_N);
        // gradientVector(prevError,currError,errorGrad,lstm_N);
        
        // gradNorm = euclideanNorm(errorGrad);
        
        // for(i=0; i < target.size();i++)
        // {
        //      prevError[i] = currError[i];
        // }
        
        //if(gradNorm < gradNormThreshold)
        //   break;
        
        ////////////////////////////////////
        ////Stopping Criteria 2 (change in MSE between epochs)
        
        // currSE = errorFunction(target,output)/target.size();
        
        // if((abs(currSE-prevSE)/currSE) < SEThreshold)
        //     break;
        
        ////////////////////////////////////////////
        ////Stopping Criteria 3 (minimum in Euclidean distance)
        ////  thus far seems to work best
        
        currSE = errorFunction(target,output);
        
        if(j==0)
        {
            minSE = currSE;
        }
        else if(minSE < currSE)
        {
            break;
        }
        else if(currSE < minSE)
        {
            minSE = currSE;
            
        }
        
        cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
        
        
        /** tempVec = lSTM_1_M->getZ_t();
         printMatrix(tempVec);
         tempVec = lSTM_1_M->getI_t();
         printMatrix(tempVec);
         tempVec = lSTM_1_M->getF_t();
         printMatrix(tempVec);
         tempVec = lSTM_1_M->getO_t();
         printMatrix(tempVec);*/
        /////////////////////////////////////////
        //Serialize each output instance
        //sprintf(filename_buf,"%sLSTM_cpp_%d.csv",outputPath,j);
        sprintf(filename_buf,"%s%d.csv",outputPath,j);
        //cout<<filename_buf;
        filebuf fb;
        fb.open(filename_buf,ios::out|ios::app);
        ostream os(&fb);
        lSTM_1_M->serialize(os);
        fb.close();
    }
    //////////////////////////////////////////////
    lSTM_1_M->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    
    
    for(i=0; i < target.size();i++)
    {
        cout<<"t = "<<i<<":\n";
        cout<<" Y:\n";
        cout<<" ";
        for(j=0; j <1;j++)
        {
            cout<<output[i][j]<<", ";
        }
        cout<<"\n";
        cout<<" Y_target: \n";
        cout<<" ";
        for(j=0; j <1;j++)
        {
            cout<<target[i][j]<<", ";
        }
        cout<<"\n";
        
    }
    
    
    
}

void LSTMTest::testSerialization()
{
    char* path = "lstm_1x1.csv";
    filebuf fb;
    fb.open(path,ios::out|ios::app);
    ostream os(&fb);
    lSTM->serialize(os);
    fb.close();
    
}

void LSTMTest::testLSTMMath()
{
    long double **A;
    long double **T;
    
    long double **A2;
    long double **T2;
    
    long double y[3];
    vector<long double> x;
    vector<long double> x2;
    int N = 3;
    int M = 4;
    int i,j;
    
    A = (long double**)malloc(sizeof(long double*)*N);
    T = (long double**)malloc(sizeof(long double*)*N);
    
    A2 = (long double**)malloc(sizeof(long double*)*N);
    T2 = (long double**)malloc(sizeof(long double*)*M);
    
    for(i = 0; i <N; i++)
    {
        A[i] = (long double*)malloc(sizeof(long double)*N);
        T[i] = (long double*)malloc(sizeof(long double)*N);
        
        A2[i] = (long double*)malloc(sizeof(long double)*M);
    }
    
    for(i = 0; i <M; i++)
    {
        T2[i] = (long double*)malloc(sizeof(long double)*N);
        
    }
    
    for(i = 0; i < N; i++)
    {
        for(j=0; j < N; j++)
        {
            if(i==j)
            {
                A[i][j] = 0.5;
                A2[i][j] = 0.5;
            }
            else
            {
                A[i][j] = 0.0;
                A2[i][j] = 0.0;
            }
        }
    }
    
    A2[0][3] = 1;
    A2[1][3] = 1;
    A2[2][3] = 1;
    
    for(i=0; i < N; i++)
    {
        x.push_back(0.5);
    }
    
    for(i=0; i < M; i++)
    {
        x2.push_back((long double)(i%2));
    }
    
    ///////////////test matrixVectorMult ///////////////////
    matrixVectorMult(A,x,y,N,N);
    
    cout<<"A Matrix\n";
    printMatrix(A,N,N);
    
    cout<<"y Vector\n";
    printVector(y,N);
    
    A[0][2] = 1;
    A[2][0] = 0.5;
    
    cout<<"A Matrix\n";
    printMatrix(A,N,N);
    transpose(A,T,N,N);
    cout<<"A Matrix Transposed\n";
    printMatrix(T,N,N);
    
    A[0][2] = 0;
    A[2][0] = 0;
    
    cout<<"A2 Matrix\n";
    printMatrix(A2,N,M);
    transpose(A2,T2,N,M);
    cout<<"A2 Matrix Transposed\n";
    printMatrix(T2,M,N);
    
    
    outerProduct(y,y,A,N,N);
    cout<<"Outer Product Matrix: y*y\n";
    printMatrix(A,N,N);
    
    cout<<"Outer Product Matrix: x*x\n";
    outerProduct(x,x,T,N,N);
    
    printMatrix(T,N,N);
    
    cout<<"Outer Product Matrix: x2*x\n";
    outerProduct(x2,x,T2,M,N);
    
    printMatrix(T2,M,N);
    
    
    for(i = 0; i < N; i++)
    {
        free(A[i]);
        free(T[i]);
        
        free(A2[i]);
    }
    
    for(i = 0; i < M; i++)
    {
        free(T2[i]);
    }
    
    free(A);
    free(T);
    
    free(A2);
    free(T2);
    
    
}

double LSTMTest::errorFunction(vector<vector<double> > target, vector<double> y)
{
    return euclideanDistance1D(target,y);
}

double LSTMTest::errorFunction(vector<vector<double> > target, vector<vector<double> >  y)
{
    return euclideanDistanceND(target,y);
}

double LSTMTest::euclideanDistance1D(vector<vector<double> > target, vector<double> y)
{
    double distance = 0.0;
    for(int i=0; i < target.size(); i++)
    {
        distance+= ((target[i][0]-y[i]))*((target[i][0]-y[i]));
    }
    return sqrt(distance);
}

double LSTMTest::euclideanNorm(vector<double> x)
{
    int i;
    double norm = 0;
    for(i=0; i < x.size(); i++)
    {
        norm +=(x[i]*x[i]);
    }
    return sqrt(norm);
}

void LSTMTest::gradientVector(vector<double> prevError, vector<double> currError, vector<double>& toRet)
{
    int i;
    toRet.clear();
    for(i=0; i < prevError.size(); i++)
    {
        toRet.push_back(currError[i]-prevError[i]);
    }
}

void LSTMTest::calcErrorVector(vector<vector<double> > target, vector<double> y, vector<double>& toRet)
{
    double error = 0.0;
    toRet.clear();
    for(int i=0; i < target.size(); i++)
    {
        error = target[i][0]-y[i];
        toRet.push_back(error);
    }
}


double LSTMTest::MSE1D(vector<vector<double> > target, vector<double> output)
{
    double MSE = 0.0;
    for(int i=0; i < target.size(); i++)
    {
        MSE+= ((target[i][0]-output[i]))*((target[i][0]-output[i]))/target.size();
        
    }
    return MSE;
}

double LSTMTest::pointWiseMaxSE(vector<vector<double> > target, vector<double> output)
{
    double maxMSE = 0.0;
    double MSE;
    for(int i=0; i < target.size(); i++)
    {
        MSE= ((target[i][0]-output[i]))*((target[i][0]-output[i]));
        if(MSE>=maxMSE)
        maxMSE=MSE;
        
    }
    return maxMSE;
}

void LSTMTest::loadTestVector1D(char* vectorPath,vector<vector<double > >& target, vector<double>& input)
{
    ifstream infile(vectorPath);
    string line;
    
    //cout<<infile.is_open()<<"\n";
    
    int i=0;
    double time_index, x,y;
    char temp;
    
    ////get input test vector to network
    while(getline(infile,line))
    {
        
        if(i!=0)
        {
            istringstream iss(line);
            iss>>time_index>>temp>>x>>temp>>y;
            //cout<<y<<"\n";
            input.push_back(x);
            vector<double> temp;
            temp.push_back(y);
            target.push_back(temp);
        }
        i++;
    }
    infile.close();
    if((target.size() < 1))
    {
        cout<<"Empty test vector!\n";
        return;
    }
    
    
    
    
    
}




void LSTMTest::testMatchToOutput()
{
    char vectorPath[] = "tests/1_1_5_pulse_gauss.csv";
    
    vector<double> output;
    vector<double> input;
    vector<vector<double > > target;
    
    int i=0;
    int j=0;
    int numIterations = 10000;
    double eta = 0.01;
    double momentum = 0.5;
    
    
    loadTestVector1D(vectorPath,target, input);
    
    double targetRef[] = {0.024048, 0.250858,0.384649, 0.255285,0.238060};
    double threshold = 0.01;
    //////////////////////////////////
    ////get input test vector to network
    
    ///////////////////////////////////////////
    //
    cout<<"Multi-iteration Test \n";
    
    lSTM->setTrainingMode(true);
    
    for(i=0; i < target.size();i++)
    {
        vector<double> temp;
        temp.push_back(input[i]);
        lSTM->setInput(temp);
        temp.clear();
        lSTM->process();
        output.push_back((lSTM->getOutput())[0]);
        cout<<output[i]<<"\n";
    }
    
    
    
    for(j=0; j < numIterations; j++)
    {
        output.clear();
        lSTM->backPropagate(eta,momentum,target);
        lSTM->setTrainingMode(false);
        lSTM->setTrainingMode(true);
        
        for(i=0; i < target.size();i++)
        {
            vector<double> temp;
            temp.push_back(input[i]);
            lSTM->setInput(temp);
            temp.clear();
            lSTM->process();
            output.push_back((lSTM->getOutput())[0]);
        }
        
        double ME = 0.0;
        
        ME= abs(((targetRef[2]-output[2]))/((targetRef[2])));
        
        
        
        if(ME < threshold)
        break;
        
        
        cout<<"Iteration: "<<j<<" Pointwise Max SE: "<<ME<<"\n";
    }
    //////////////////////////////////////////////
    lSTM->setTrainingMode(false);
    cout<<"__________________________________ \n";
    cout<<"End Network Output \n";
    cout<<" t | Y | Y_targetRef | error_t \n";
    
    for(i=0; i < target.size();i++)
    {
        cout<<i<<"|"<<output[i]<<"|"<<targetRef[i]<<"|"<<(targetRef[i]-output[i])/targetRef[i]<<"\n";
    }
    
    
}
/**
 void newtestclass::testLSTM_modY(){
 char vectorPath[] = "tests/1_1_5_pulse_gauss.csv";
 
 vector<double> output;
 vector<double> input;
 vector<vector<double > > target;
 
 int i=0;
 int j=0;
 double eta = 0.01;
 double momentum = 0.5;
 
 loadTestVector1D(vectorPath,target, input);
 
 ///////////////////////////////////////////
 //
 cout<<"Pass 0 \n";
 cout<<"Output from network: \n";
 
 lSTM_modY->setTrainingMode(true);
 
 for(i=0; i < target.size();i++)
 {
 lSTM_modY->setInput(&(input[i]));
 lSTM_modY->process();
 output.push_back(*(lSTM_modY->getOutput()));
 cout<<output[i]<<"\n";
 }
 
 output.clear();
 
 ////////////////////////////////////////////
 cout<<"Back Propagation \n";
 lSTM_modY->backPropagate(eta,momentum,target);
 lSTM_modY->setTrainingMode(false);
 lSTM_modY->setTrainingMode(true);
 
 
 /////////////////////////////////////////////
 cout<<"Pass 1 \n";
 cout<<"Output from network: \n";
 for(i=0; i < target.size();i++)
 {
 lSTM_modY->setInput(&(input[i]));
 lSTM_modY->process();
 output.push_back(*(lSTM_modY->getOutput()));
 cout<<output[i]<<"\n";
 }
 output.clear();
 
 //////////////////////////////////////////////////
 cout<<"Back Propagation \n";
 lSTM_modY->backPropagate(eta,momentum,target);
 lSTM_modY->setTrainingMode(false);
 
 
 ////////////////////////////////////////////////
 cout<<"Pass 2 \n";
 cout<<"Output from network: \n";
 for(i=0; i < target.size();i++)
 {
 lSTM_modY->setInput(&(input[i]));
 lSTM_modY->process();
 output.push_back(*(lSTM_modY->getOutput()));
 cout<<output[i]<<"\n";
 }
 output.clear();
 
 
 ///////////////////////////////////////////////
 CPPUNIT_ASSERT(true);
 
 
 
 }
 
 void newtestclass::testLSTM_modYIterations()
 {
 char vectorPath[] = "tests/1_1_5_pulse_gauss.csv";
 
 vector<double> output;
 vector<double> input;
 vector<vector<double > > target;
 
 int i=0;
 int j=0;
 int numIterations = 10000;
 
 double eta = 0.01;
 double momentum = 0.5;
 
 loadTestVector1D(vectorPath,target, input);
 
 //////////////////////////////////
 ///////////////////////////////////////////
 //
 cout<<"Multi-iteration Test \n";
 
 lSTM_modY->setTrainingMode(true);
 double minSE,currSE = 0;
 
 for(i=0; i < target.size();i++)
 {
 lSTM_modY->setInput(&(input[i]));
 lSTM_modY->process();
 output.push_back(*(lSTM_modY->getOutput()));
 cout<<output[i]<<"\n";
 }
 
 
 
 for(j=0; j < numIterations; j++)
 {
 output.clear();
 lSTM_modY->backPropagate(eta,momentum,target);
 lSTM_modY->setTrainingMode(false);
 lSTM_modY->setTrainingMode(true);
 
 for(i=0; i < target.size();i++)
 {
 lSTM_modY->setInput(&(input[i]));
 lSTM_modY->process();
 output.push_back(*(lSTM_modY->getOutput()));
 }
 
 currSE = errorFunction(target,output);
 
 if(j==0)
 {
 minSE = currSE;
 }
 else if(minSE < currSE)
 {
 break;
 }
 else if(currSE < minSE)
 {
 minSE = currSE;
 
 }
 
 cout<<"Iteration: "<<j<<" Euclidean Distance: "<<currSE<<"\n";
 }
 //////////////////////////////////////////////
 lSTM_modY->setTrainingMode(false);
 cout<<"__________________________________ \n";
 cout<<"End Network Output \n";
 cout<<" t | Y | Y_target | error_t \n";
 
 for(i=0; i < target.size();i++)
 {
 cout<<i<<"|"<<output[i]<<"|"<<target[i][0]<<"|"<<(target[i][0]-output[i])/target[i][0]<<"\n";
 }
 
 
 ///////////////////////////////////////////////
 CPPUNIT_ASSERT(true);
 
 }
 */

void LSTMTest::printMatrix(long double** A, int N, int M)
{
    int i,j;
    
    cout<<"____________________\n";
    cout<<"[";
    for(i = 0 ; i < N; i++)
    {
        
        for(j = 0; j < M; j++)
        {
            cout<<A[i][j]<<" ";
        }
        cout<<",\n";
    }
    cout<<"]\n";
}

void LSTMTest::printMatrix(vector<vector<long double> > A)
{
    int i,j;
    
    cout<<"____________________\n";
    cout<<"[";
    for(i = 0 ; i < A.size(); i++)
    {
        
        for(j = 0; j < A[i].size(); j++)
        {
            cout<<A[i][j]<<" ";
        }
        cout<<",\n";
    }
    cout<<"]\n";
    
}

void LSTMTest::printVector(vector<long double> y, int N)
{
    
    cout<<"_______________________\n";
    cout<<"[";
    for(int i = 0;i<N; i++)
    {
        cout<<y[i]<<",";
    }
    cout<<"]\n'";
}

void LSTMTest::printVector(long double* y, int N)
{
    
    cout<<"_______________________\n";
    cout<<"[";
    for(int i = 0;i<N; i++)
    {
        cout<<y[i]<<",";
    }
    cout<<"]\n'";
}

void LSTMTest::loadTestVectorND(char* vectorPath,vector<vector<double > >& target, vector<vector<double > >& input, int N, int M)
{
    ifstream infile(vectorPath);
    string line;
    
    //cout<<infile.is_open()<<"\n";
    
    int i=0;
    double time_index, x,y;
    char temp;
    
    ////get input test vector to network
    while(getline(infile,line))
    {
        
        if(i!=0)
        {
            istringstream iss(line);
            iss>>time_index;
            vector<double> temp_X_i;
            for(int j=0; j< M; j++)
            {
                iss>>temp>>x;
                temp_X_i.push_back(x);
            }
            input.push_back(temp_X_i);
            
            vector<double> temp_Y_i;
            
            for(int j=0; j< N; j++)
            {
                iss>>temp>>y;
                temp_Y_i.push_back(y);
            }
            target.push_back(temp_Y_i);
        }
        i++;
    }
    infile.close();
    if((target.size() < 1))
    {
        cout<<"Empty test vector!\n";
        return;
    }
    
    
    
    
}

double LSTMTest::euclideanDistanceND(vector<vector<double> > target, vector<vector<double> >  y)
{
    
    double mean_distance = 0.0;
    for(int i=0; i < target.size(); i++)
    {
        double distance = 0.0;
        for(int j = 0; j<target[0].size(); j++ )
        {
            distance+= ((target[i][j]-y[i][j]))*((target[i][j]-y[i][j]));
            distance = sqrt(distance);
        }
        mean_distance = distance/target.size();
        
    }
    return mean_distance;
    
}
