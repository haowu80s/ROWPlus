//
// Created by Hao Wu on 11/7/16.
//

#include <iostream>
#include "ROWPlus/ROWPlus.h"
#include "lorenz96.h"
#include <boost/numeric/odeint.hpp>

using namespace Eigen;
using namespace ROWPlus;
using namespace boost::numeric::odeint;
using namespace std;

template<typename vec>
void initX(vec& x, size_t N) {
  for (int i = 0; i < N; ++i)
    x[i] = 0.5 * ((double) i - 0.5 * (double) N);
}

int main() {
  const double t1 = 3.0;

  lorenz96 fun;

  std::cout << "Lorenz-96 (N=40)" << std::endl;

  lorenz96::state_type x_ref(40);
  initX(x_ref, 40);
  typedef runge_kutta_cash_karp54<lorenz96::state_type> error_stepper_type;
  integrate_adaptive(make_controlled<error_stepper_type>(1.0e-12, 1.0e-12),
                     fun, x_ref, 0.0, t1, 1e-8 );

  // initialize solution vectors
  lorenz96::state_type x_t1(40);
  VectorXd x_t2(40);

  // creat solver: runge_kutta4_classic
  runge_kutta4_classic< lorenz96::state_type > stepper_rk4;
  // create solver: ROWPlus::rosenbrock4
  ROWPlus::rosenbrock4<lorenz96, double> stepper_grk4t;
  stepper_grk4t.makeConstantStepper(&fun);
  // creat solver: rosenbrock_krylov4 ROK4A
  rosenbrock_krylov4<lorenz96, double> stepper_rok4a(4);
  auto _opts = stepper_rok4a.getOptions();
  _opts.TypeScheme = ROK4A;
  stepper_rok4a.setOptions(_opts);
  stepper_rok4a.makeConstantStepper(&fun);
  // creat solver: rosenbrock_krylov4 ROK4E
  rosenbrock_krylov4<lorenz96, double> stepper_rok4e(4);
  stepper_rok4e.makeConstantStepper(&fun);

  //
  double dt = 0.05;
  while (dt >= 5e-5) {
    ROWPlusSolverSpace::Status ret;
    cout << dt << " ";
    // stepper_rk54
    initX(x_t1, 40);
    integrate_const( stepper_rk4 , fun , x_t1 , 0.0 , t1 , dt );
    Map<VectorXd>(x_t1.data(), 40) -= Map<VectorXd>(x_ref.data(), 40);
    cout << Map<VectorXd>(x_t1.data(), 40).stableNorm() /
            Map<VectorXd>(x_ref.data(), 40).stableNorm() << " ";
    // stepper_grk4t
    initX(x_t2, 40);
    ret = stepper_grk4t.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), 40);
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), 40).stableNorm() << " ";
    // stepper_rok4a
    initX(x_t2, 40);
    ret = stepper_rok4a.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), 40);
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), 40).stableNorm() << " ";
    // stepper_rok4e
    initX(x_t2, 40);
    ret = stepper_rok4e.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), 40);
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), 40).stableNorm() << " ";

    cout << endl;
    dt /= 2.0;
  }
  return 0;
}