//
// Created by Hao Wu on 11/7/16.
//

#include "lorenz96.h"

using namespace Eigen;
using namespace ROWPlus;

typedef BaseFunctor<double, 40, 40> BF;

lorenz96::lorenz96(double F)
    : m_F(F), m_wa1(InputType(inputs())), m_wa2(inputs()) {
  eigen_assert(inputs() == values());
};

void lorenz96::operator()(const lorenz96::state_type &x,
                          lorenz96::state_type &dxdt,
                          const double _) {
  for (int j = 2; j < inputs() - 1; ++j) {
    dxdt[j] = -x[j - 1] * (x[j - 2] - x[j + 1]) - x[j] + m_F;
  }
  dxdt[0] = -x[inputs() - 1] * (x[inputs() - 2] - x[1]) - x[0] + m_F;
  dxdt[1] = -x[0] * (x[inputs() - 1] - x[2]) - x[1] + m_F;
  dxdt[inputs() - 1] =
      -x[inputs() - 2] * (x[inputs() - 3] - x[0]) - x[inputs() - 1] + m_F;
}

int lorenz96::f(double t, const Ref<const VectorXd> x, Ref<VectorXd> v) {
  for (int j = 2; j < inputs() - 1; ++j) {
    v[j] = -x[j - 1] * (x[j - 2] - x[j + 1]) - x[j] + m_F;
  }
  v[0] = -x[inputs() - 1] * (x[inputs() - 2] - x[1]) - x[0] + m_F;
  v[1] = -x[0] * (x[inputs() - 1] - x[2]) - x[1] + m_F;
  v[inputs() - 1] =
      -x[inputs() - 2] * (x[inputs() - 3] - x[0]) - x[inputs() - 1] + m_F;
  return 0;
};

int lorenz96::fdt(double t, const Ref<const VectorXd> x,
                  Ref<VectorXd> v) {
  throw std::runtime_error("lorenz96::fdt() is not implemented.");
  return -1;
};

int lorenz96::df(double t,
                 const Ref<const VectorXd> x,
                 Ref<MatrixXd> m) {
  throw std::runtime_error("lorenz96::df() is not implemented.");
  return -1;
};
