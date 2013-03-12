#pragma once

#include <Eigen/Dense>

namespace OpenANN
{

enum ActivationFunction
{
  /**
   * Logistic sigmoid activation function.
   *
   * Range: [0, 1].
   *
   * \f$ g(a) = \frac{1}{1+\exp{{-a}}} \f$
   */
  LOGISTIC,
  /**
   * Tanh sigmoid function.
   *
   * Range: [-1, 1].
   *
   * \f$ g(a) = tanh(a) \f$
   */
  TANH,
  /**
   * Scaled tanh sigmoid function.
   *
   * Range: [-1.7159, 1.7159].
   *
   * This activation function does not saturate around the output -1 or +1.
   *
   * \f$ g(a) = 1.7159 tanh(\frac{2}{3} a) \f$
   */
  TANH_SCALED,
  /**
   * Biologically inspired, non-saturating rectified linear unit (ReLU).
   *
   * Range: [0, \f$ \infty \f$].
   *
   * \f$ g(a) = max(0, a) \f$
   *
   * [1] X. Glorot, A. Bordes, Y. Bengio:
   * Deep Sparse Rectifier Neural Networks,
   * International Conference on Artificial Intelligence and Statistics 15,
   * pp. 315–323, 2011.
   */
  RECTIFIER,
  /**
   * Identity function.
   *
   * Range: [\f$ -\infty \f$, \f$ \infty \f$].
   *
   * \f$ g(a) = a \f$
   */
  LINEAR
};

void activationFunction(ActivationFunction act, const Vt& a, Vt& z);
void activationFunctionDerivative(ActivationFunction act, const Vt& z, Vt& gd);

void softmax(Vt& y);
void logistic(const fpt* a, fpt* z, const int J);
void logisticDerivative(const fpt* z, fpt* gd, const int J);
void normaltanh(const fpt* a, fpt* z, const int J);
void normaltanhDerivative(const fpt* z, fpt* gd, const int J);
void scaledtanh(const fpt* a, fpt* z, const int J);
void scaledtanhDerivative(const fpt* z, fpt* gd, const int J);
void rectifier(const fpt* a, fpt* z, const int J);
void rectifierDerivative(const fpt* z, fpt* gd, const int J);
void linear(const fpt* a, fpt* z, const int J);
void linearDerivative(fpt* gd, const int J);

}
