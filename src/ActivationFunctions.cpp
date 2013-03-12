#include <ActivationFunctions.h>
#include <limits>
#include <cmath>
#include <cstring>

namespace OpenANN
{

void activationFunction(ActivationFunction act, const Vt& a, Vt& z)
{
  switch(act)
  {
    case LOGISTIC:
      logistic(a.data(), z.data(), a.rows());
      break;
    case TANH:
      normaltanh(a.data(), z.data(), a.rows());
      break;
    case TANH_SCALED:
      scaledtanh(a.data(), z.data(), a.rows());
      break;
    case RECTIFIER:
      rectifier(a.data(), z.data(), a.rows());
      break;
    case LINEAR:
    default:
      linear(a.data(), z.data(), a.rows());
      break;
  }
}

void activationFunctionDerivative(ActivationFunction act, const Vt& z, Vt& gd)
{
  switch(act)
  {
    case LOGISTIC:
      logisticDerivative(z.data(), gd.data(), gd.rows());
      break;
    case TANH:
      normaltanhDerivative(z.data(), gd.data(), gd.rows());
      break;
    case TANH_SCALED:
      scaledtanhDerivative(z.data(), gd.data(), gd.rows());
      break;
    case RECTIFIER:
      rectifierDerivative(z.data(), gd.data(), gd.rows());
      break;
    case LINEAR:
    default:
      linearDerivative(gd.data(), gd.rows());
      break;
  }
}

void softmax(Vt& y)
{
  const int F = y.rows();
  const fpt max = y.maxCoeff();
  for(int f = 0; f < F; f++)
    y(f) = std::exp(y(f) - max);
  y /= y.sum();
}

void logistic(const fpt * a, fpt* z, const int J)
{
  for(int j = 0; j < J; j++, a++, z++)
  {
    if(*a < -45.0)
      *z = 0.0;
    else if(*a > 45.0)
      *z = 1.0;
    else
      *z = 1.0 / (1.0+std::exp(-(*a)));
  }
}

void logisticDerivative(const fpt* z, fpt* gd, const int J)
{
  for(int j = 0; j < J; j++, z++, gd++)
  {
    register fpt zv = *z;
    *gd = zv*(1.0 - zv);
  }
}

void normaltanh(const fpt* a, fpt* z, const int J)
{
  for(int j = 0; j < J; j++, a++, z++)
    *z = std::tanh(*a);
}

void normaltanhDerivative(const fpt* z, fpt* gd, const int J)
{
  for(int j = 0; j < J; j++, z++, gd++)
  {
    register fpt zv = *z;
    *gd = 1.0 - zv*zv;
  }
}

void scaledtanh(const fpt* a, fpt* z, const int J)
{
  for(int j = 0; j < J; j++, a++, z++)
    *z = 1.7159*std::tanh(0.66666667 * *a);
}

void scaledtanhDerivative(const fpt* z, fpt* gd, const int J)
{
  for(int j = 0; j < J; j++, z++, gd++)
  {
    register fpt zv = 1.7159 * *z;
    *gd = 0.66666667/1.7159 * zv*zv;
  }
}

void rectifier(const fpt* a, fpt* z, const int J)
{
  for(int j = 0; j < J; j++, a++, z++)
    *z = std::max<fpt>(0.0, *a);
}

void rectifierDerivative(const fpt* z, fpt* gd, const int J)
{
  for(int j = 0; j < J; j++, gd++, z++)
    *gd = *z == (fpt) 0.0 ? 0.0 : 1.0;
}

void linear(const fpt* a, fpt* z, const int J)
{
  std::memcpy(z, a, sizeof(fpt)*J);
}

void linearDerivative(fpt* gd, const int J)
{
  for(int j = 0; j < J; j++, gd++)
    *gd = 1.0;
}

}
