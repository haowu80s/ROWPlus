//
// Created by Hao Wu on 11/7/16.
//

#include <iostream>
#include "lorenz96.h"
#include <boost/numeric/odeint.hpp>

using namespace Eigen;
using namespace ROWPlus;
using namespace boost::numeric::odeint;

int main() {
  const double aTol = 1.0e-12;
  const double rTol = 1.0e-4;
  const double t1 = 0.3;
  std::srand(101);
  lorenz96 fun;
  VectorXd x_1(40);
  for (int i = 0; i < 40; ++i) {
    x_1[i] = (double) (i - 20);
  }
  std::cout << "Lorenz-96 (N=40)" << std::endl;
  ODEOptions<double> opts;
  opts.relTol = rTol;
  opts.absTol = aTol;
  opts.h_max = 1e-2;
  opts.h_init = 1e-5;
  opts.TypeScheme = ROK4L;
  opts.iUserJac = false;
  opts.maxKryDim = 4;
  opts.minKryDim = 4;
  opts.maxJacReuse = 0;
  opts.maxSteps = 1000000;

  ODESolver<ROWPlus::ODEJacSAP<lorenz96>, lorenz96> solver(&fun, opts);
  ROWPlusSolverSpace::Status ret = solver.step(x_1, 0.0, t1);
  solver.getStats().print();
  assert(ret == ROWPlusSolverSpace::ComputeSucessful);

  lorenz96::state_type x_2(40);
  for (int i = 0; i < 40; ++i) {
    x_2[i] = (double) (i - 20);
  }

  typedef runge_kutta_cash_karp54<lorenz96::state_type> error_stepper_type;
  integrate_adaptive(make_controlled<error_stepper_type>(1.0e-12, 1.0e-12),
                     fun, x_2, 0.0, t1, 1e-5 );
  x_1 -= Map<VectorXd>(x_2.data(), 40);
  cout << "error = "
       << x_1.stableNorm()
       << "/"
       << Map<VectorXd>(x_2.data(), 40).stableNorm();
  cout << " = " << x_1.stableNorm() / Map<VectorXd>(x_2.data(), 40).stableNorm()
       << endl;
  return 0;
}