//
//  main.cpp
//  LSTM
//
//  Created by Stuart Truax on 9/10/18.
//  Copyright © 2018 Stuart Truax. All rights reserved.
//

#include <iostream>
#include "LSTMTest.h"

int main(int argc, const char * argv[]) {
    
    LSTMTest* t =  new LSTMTest();
    
    cout<<"Begin Unit Tests of LSTM Networks"<<"\n"; 
    
    t->setUp();
    t->testIterations();
    t->testIterations_N_N();
    t->testIterations_N_M();
    t->testIterations_N_1();
    t->testIterations_1_M();
    t->testLSTMMath();
    t->tearDown();
    
    return 0;
}
