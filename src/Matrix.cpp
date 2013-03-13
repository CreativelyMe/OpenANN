#include <utils/Matrix.h>
#include <OpenANNException.h>
//#undef CUDA_AVAILABLE
#if CUDA_AVAILABLE
#include <cuBLASInterface.cuh>
#endif

namespace OpenANN
{

Matrix::Matrix()
    : rawAllocated(false), rawCurrent(false),
      cudaAllocated(false), cudaCurrent(false),
      eigenAllocated(false), eigenCurrent(false),
      initialized(false), rows(-1), cols(-1), raw(0), cuda(0), eigen(0)
{
}

Matrix::~Matrix()
{
  if(rawAllocated)
    delete[] raw;
#ifdef CUDA_AVAILABLE
  if(cudaAllocated)
    CUBLASContext::instance.freeMatrix(cuda);
#endif
  if(eigenAllocated)
    delete eigen;
}

void Matrix::wrapEigen(Eigen::Matrix<fpt, Eigen::Dynamic, Eigen::Dynamic>* eigen, bool cleanUp)
{
  if(initialized)
    throw OpenANNException("Matrix is already initialized.");

  this->eigen = eigen;
  this->rows = eigen->rows();
  this->cols = eigen->cols();
  size = rows * cols;
  eigenAllocated = cleanUp;
  eigenCurrent = true;

  raw = eigen->data();
  rawAllocated = false;
  rawCurrent = true;

#if CUDA_AVAILABLE
  CUBLASContext::instance.allocateMatrix(&cuda, rows, cols);
  CUBLASContext::instance.setMatrix(raw, cuda, rows, cols);
  cudaAllocated = true;
  cudaCurrent = true;
#else
  cudaAllocated = false;
  cudaCurrent = false;
#endif

  initialized = true;
}

bool Matrix::sane()
{
  if(rawCurrent)
  {
    const fpt* const end = raw + size;
    for(const fpt* p = raw; p < end; p++)
      if(isnan(*p) || isinf(*p))
        return false;
  }
  else if(cudaCurrent)
    return true; // TODO check for inf and nan
  else
    return true;
}

void Matrix::gemm(const Matrix& in, Matrix& out)
{
#if CUDA_AVAILABLE
  CUBLASContext::instance.multiplyMatrixMatrix(cuda, in.cuda, out.cuda, rows, cols, in.cols);
#else
  *out.eigen = *eigen * *in.eigen;
#endif
}

}
