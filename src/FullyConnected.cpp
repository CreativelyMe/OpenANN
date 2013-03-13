#include <layers/FullyConnected.h>
#include <Random.h>
#undef CUDA_AVAILABLE
#ifdef CUDA_AVAILABLE
#include <cuBLASInterface.cuh>
#endif

namespace OpenANN {

FullyConnected::FullyConnected(OutputInfo info, int J, bool bias,
                               ActivationFunction act, fpt stdDev,
                               fpt dropoutProbability,
                               fpt maxSquaredWeightNorm)
  : I(info.outputs()), J(J), bias(bias), act(act), stdDev(stdDev),
    dropoutProbability(dropoutProbability),
    maxSquaredWeightNorm(maxSquaredWeightNorm), W(J, I), Wd(J, I), x(0), a(J),
    y(J+bias), yd(J), deltas(J), e(I)
{
#ifdef CUDA_AVAILABLE
  if(!CUBLASContext::instance.allocateMatrix(&WOnDevice, J, I))
    abort();
  if(!CUBLASContext::instance.allocateMatrix(&xOnDevice, I, 1))
    abort();
  if(!CUBLASContext::instance.allocateMatrix(&aOnDevice, J, 1))
    abort();
#endif
}

FullyConnected::~FullyConnected()
{
#ifdef CUDA_AVAILABLE
  if(!CUBLASContext::instance.freeMatrix(WOnDevice))
    abort();
  if(!CUBLASContext::instance.freeMatrix(xOnDevice))
    abort();
  if(!CUBLASContext::instance.freeMatrix(aOnDevice))
    abort();
#endif
}

OutputInfo FullyConnected::initialize(std::vector<fpt*>& parameterPointers,
                                      std::vector<fpt*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + J*I);
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J*I);
  for(int j = 0; j < J; j++)
  {
    for(int i = 0; i < I; i++)
    {
      parameterPointers.push_back(&W(j, i));
      parameterDerivativePointers.push_back(&Wd(j, i));
    }
  }

  // Bias component will not change after initialization
  if(bias)
    y(J) = fpt(1.0);

  initializeParameters();

  OutputInfo info;
  info.bias = bias;
  info.dimensions.push_back(J);
  return info;
}

void FullyConnected::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I; i++)
      W(j, i) = rng.sampleNormalDistribution<fpt>() * stdDev;
}

void FullyConnected::updatedParameters()
{
  if(maxSquaredWeightNorm > 0.0)
  {
    for(int j = 0; j < J; j++)
    {
      const fpt squaredNorm = W.row(j).squaredNorm();
      if(squaredNorm > maxSquaredWeightNorm)
        W.row(j) *= sqrt(maxSquaredWeightNorm / squaredNorm);
    }
  }
#ifdef CUDA_AVAILABLE
  if(!CUBLASContext::instance.setMatrix(W.data(), WOnDevice, J, I))
    abort();
#endif
}

void FullyConnected::forwardPropagate(Vt* x, Vt*& y, bool dropout)
{
  this->x = x;
  // Activate neurons
#ifdef CUDA_AVAILABLE
  if(!CUBLASContext::instance.setMatrix(x->data(), xOnDevice, I, 1))
    abort();
  if(!CUBLASContext::instance.multiplyMatrixVector(WOnDevice, xOnDevice, aOnDevice, J, I))
    abort();
  if(!CUBLASContext::instance.getMatrix(a.data(), aOnDevice, J, 1))
    abort();
#else
  a = W * *x;
#endif
  // Compute output
  activationFunction(act, a, this->y);
  if(dropout)
  {
    RandomNumberGenerator rng;
    for(int j = 0; j < J; j++)
      if(rng.generate<fpt>(0.0, 1.0) < dropoutProbability)
        this->y(j) = (fpt) 0;
  }
  else if(dropoutProbability > 0.0)
  {
    // Hinton et al., 2012: "At test time, we use the "mean network" [...] to
    // compensate for the fact that [all] of them are active."
    // Scaling the outputs is equivalent to scaling the outgoing weights.
    this->y *= (1.0 - dropoutProbability);
    if(bias)
      this->y(J) = 1.0;
  }
  y = &(this->y);
}

void FullyConnected::backpropagate(Vt* ein, Vt*& eout)
{
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  for(int j = 0; j < J; j++)
    deltas(j) = yd(j) * (*ein)(j);
  // Weight derivatives
  Wd = deltas * x->transpose();
  // Prepare error signals for previous layer
  e = W.transpose() * deltas;
  eout = &e;
}

Vt& FullyConnected::getOutput()
{
  return y;
}

}
