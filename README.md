# LSTM

A C++ implementation of J. Schmidhuber’s LSTM recurrent neural network architecture for sequential data. 

## Getting Started

### Prerequisites

A. XCode 9.2 or later
B. gcc or any C++ compiler with STL support


### Installing

Build the project in Xcode from the LSTM.xcodeproj file

## Running the tests

The file main.cpp instantiates the LSTMTest class. The LSTM class is the implementation of the LSTM network architecture proper. 

Several methods in the LSTMTest class create instances of the LSTM class, with varying input and output dimensions between the instances. 

The testing methods in LSTMTest use temporal input vectors formatted as .csv files, which are located in the Debug/tests directory. These input vectors are used to train the LSTM networks in each of the unit tests within the LSTMTest class. 


## Built With

* Xcode 9.2


##References

[1]  K. Greff, R. K. Srivastava, J. Koutnik, B.R. Steunebrink, J. Schmidhuber 
     "LSTM: a search space odyssey", arXiv, 2015

## Authors

* **Stuart Truax** - *Initial work* - (https://github.com/StuartTruax)


## License





