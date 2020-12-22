//g++ -o test test.cpp -I../ -I./ -L../../build/nntrainer -lnntrainer -I../../nntrainer/tensor -DEIGEN_USE_THREADS -pthread -std=c++14 -w -mavx

#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <iomanip>
#include <thread>
#include <tensor.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/NeuralNetworks/SpatialConvolutions.h>
#include <unsupported/Eigen/CXX11/ThreadPool>

typedef Eigen::Tensor<float, 1, Eigen::ColMajor> tensor1d_t; ///< 1D tensor of float type
typedef Eigen::Tensor<float, 2, Eigen::ColMajor> tensor2d_t; ///< 2D tensor of float type
typedef Eigen::Tensor<float, 3, Eigen::ColMajor> tensor3d_t; ///< 3D tensor of float type
typedef Eigen::Tensor<float, 4, Eigen::ColMajor> tensor4d_t; ///< 4D tensor of float type
typedef Eigen::Tensor<float, 5, Eigen::ColMajor> tensor5d_t; ///< 5D tensor of float type
typedef Eigen::Tensor<float, 6, Eigen::ColMajor> tensor6d_t; ///< 6D tensor of float type
typedef Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
  EigenTensor;
  
template<typename Scalar, typename... Dims>
auto Scalar_to_Tensor(const Scalar *matrix, Dims... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix, {dims...});
}

struct point {
    double a;
    double b;
};
 
void EigenMapExample()
{
    ////////////////////////////////////////First Example/////////////////////////////////////////
    Eigen::VectorXd solutionVec(12,1);
    solutionVec<<1,2,3,4,5,6,7,8,9,10,11,12;
    Eigen::Map<Eigen::MatrixXd> solutionColMajor(solutionVec.data(),4,3);
 
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> >solutionRowMajor (solutionVec.data());
 
 
    std::cout << "solutionColMajor: "<< std::endl;
    std::cout << solutionColMajor<< std::endl;
 
    std::cout << "solutionRowMajor"<< std::endl;
    std::cout << solutionRowMajor<< std::endl;
 
    ////////////////////////////////////////Second Example/////////////////////////////////////////
 
    int array[9];
    for (int i = 0; i < 9; ++i) {
        array[i] = i;
    }
 
    Eigen::MatrixXi a(9, 1);
    a = Eigen::Map<Eigen::Matrix3i>(array);
    std::cout << a << std::endl;
 
    std::vector<point> pointsVec;
    point point1, point2, point3;
 
    point1.a = 1.0;
    point1.b = 1.5;
 
    point2.a = 2.4;
    point2.b = 3.5;
 
    point3.a = -1.3;
    point3.b = 2.4;
 
    pointsVec.push_back(point1);
    pointsVec.push_back(point2);
    pointsVec.push_back(point3);
 
    Eigen::Matrix2Xd pointsMatrix2d = Eigen::Map<Eigen::Matrix2Xd>(
        reinterpret_cast<double*>(pointsVec.data()), 2,  long(pointsVec.size()));
 
    Eigen::MatrixXd pointsMatrixXd = Eigen::Map<Eigen::MatrixXd>(
        reinterpret_cast<double*>(pointsVec.data()), 2, long(pointsVec.size()));
 
    std::cout << pointsMatrix2d << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << pointsMatrixXd << std::endl;
    std::cout << "==============================" << std::endl;
 
    std::vector<Eigen::Vector3d> eigenPointsVec;
    eigenPointsVec.push_back(Eigen::Vector3d(2, 4, 1));
    eigenPointsVec.push_back(Eigen::Vector3d(7, 3, 9));
    eigenPointsVec.push_back(Eigen::Vector3d(6, 1, -1));
    eigenPointsVec.push_back(Eigen::Vector3d(-6, 9, 8));
 
    Eigen::MatrixXd pointsMatrix = Eigen::Map<Eigen::MatrixXd>(eigenPointsVec[0].data(), 3, long(eigenPointsVec.size()));
 
    std::cout << pointsMatrix << std::endl;
    std::cout << "==============================" << std::endl;
 
    pointsMatrix = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double*>(eigenPointsVec.data()), 3, long(eigenPointsVec.size()));
 
    std::cout << pointsMatrix << std::endl;
 
    std::vector<double> aa = { 1, 2, 3, 4 };
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aa.data(), long(aa.size()));

    float ab[28*28*3];
    float kb[3*3*3*5];
    float bb[28*28*5];    

    for(int i=0;i<28*28*3;++i){
      ab[i]=1.0;
    }

    for(int i=0;i<3*3*3*5;++i){
      kb[i]=2.0;
    }
    nntrainer::TensorDim a_dim(1,3,28,28);
    nntrainer::TensorDim k_dim(5,3,3,3);
    nntrainer::TensorDim b_dim(1,5,28,28);
    
    nntrainer::Tensor ta = nntrainer::Tensor( a_dim, ab);
    nntrainer::Tensor tk = nntrainer::Tensor( k_dim, kb);

    
    Eigen::Tensor<float,3> ifm = Scalar_to_Tensor(ta.getData(),3,28,28);
    Eigen::Tensor<float,4> kernels = Scalar_to_Tensor(tk.getData(),5,3,3,3);

    const Eigen::Tensor<float,3> dest=Eigen::SpatialConvolution(ifm, kernels, 1, 1, Eigen::PADDING_VALID);
    std::cout << dest << std::endl;

    nntrainer::Tensor tb = nntrainer::Tensor( b_dim, dest.data());
    std::cout << tb << std::endl;
    
}

int main(){
  EigenMapExample();
  return 0;
}

