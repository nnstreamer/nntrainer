#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include "include/matrix.h"
#include "include/neuralnet.h"
#define training false

std::string data_file;

double stepFunction(double x) {
  if (x > 0.5) {
    return 1.0;
  }

  if (x < 0.5) {
    return 0.0;
  }

  return x;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "./LogisticRegression Config.ini data.txt\n";
    exit(0);
  }

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_file = args[1];

  srand(time(NULL));

  std::vector<std::vector<double>> inputVector, outputVector;
  Network::NeuralNetwork NN(config);

  NN.init();
  if (!training)
    NN.readModel();

  std::ifstream dataFile(data_file);
  if (dataFile.is_open()) {
    std::string temp;
    int index = 0;
    while (std::getline(dataFile, temp)) {
      if (training && index % 10 == 1) {
        std::cout << temp << std::endl;
        index++;
        continue;
      }
      std::istringstream buffer(temp);
      std::vector<double> line;
      std::vector<double> out;
      double x;
      for (int i = 0; i < 2; i++) {
        buffer >> x;
        line.push_back(x);
      }
      inputVector.push_back(line);
      buffer >> x;
      out.push_back(x);
      outputVector.push_back(out);
      index++;
    }
  }
  if (training) {
    for (unsigned int i = 0; i < NN.getEpoch(); i++) {
      NN.backwarding(Matrix(inputVector), Matrix(outputVector), i);
      std::cout << "#" << i + 1 << "/" << NN.getEpoch()
                << " - Loss : " << NN.getLoss() << std::endl;
      NN.setLoss(0.0);
    }
  } else {
    std::cout << NN.forwarding(Matrix(inputVector)).applyFunction(stepFunction)
              << std::endl;
  }

  NN.saveModel();
  NN.finalize();
}
