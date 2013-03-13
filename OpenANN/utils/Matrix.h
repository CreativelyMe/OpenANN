#pragma once

#include <Eigen/Dense>

namespace OpenANN
{

class Matrix
{
  bool rawAllocated, rawCurrent;
  bool cudaAllocated, cudaCurrent;
  bool eigenAllocated, eigenCurrent;
public:
  bool initialized;
  int rows, cols;
  int size;
  fpt* raw;
  fpt* cuda;
  Eigen::Matrix<fpt, Eigen::Dynamic, Eigen::Dynamic>* eigen;

  Matrix();
  ~Matrix();

  void wrapEigen(Eigen::Matrix<fpt, Eigen::Dynamic, Eigen::Dynamic>* eigen, bool cleanUp=false);

  bool sane();

  void gemm(const Matrix& in, Matrix& out);

private:
  void copyOnDevice();
};

}