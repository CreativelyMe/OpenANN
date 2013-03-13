#include <utils/Matrix.h>
#include <io/Logger.h>
#include <Eigen/Dense>

using namespace OpenANN;

int main()
{
  Logger logger(Logger::CONSOLE);

  int M = 7000, N = 7000, K = 7000;
  Mt A(M, K);
  A.fill(2.0);
  Matrix a;
  a.wrapEigen(&A);
  logger << a.rows << " x " << a.cols << "\n";

  Mt B(K, N);
  B.fill(1.0);
  Matrix b;
  b.wrapEigen(&B);
  logger << b.rows << " x " << b.cols << "\n";

  Mt C(M, N);
  Matrix c;
  c.wrapEigen(&C);
  logger << c.rows << " x " << c.cols << "\n";

  if(a.sane())
    logger << "sane\n";
  else
    logger << "not sane\n";

  a.gemm(b, c);
  
}
