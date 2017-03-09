//
// Created by Hao Wu on 11/10/16.
//

//
// Created by Hao Wu on 11/7/16.
//

#include <iostream>
#include "ROWPlus/ROWPlus.h"
#include <boost/numeric/odeint.hpp>

#include "RxnFunctor.h"

using namespace Eigen;
using namespace ROWPlus;
using namespace boost::numeric::odeint;
using namespace Cantera;
using namespace std;

template<typename vec>
void initX(vec& x, IdealGasMix& gas) {
  x.resize(gas.nSpecies()+1);
  gas.setState_TPX(300.0, OneAtm,
                   "H:1e-3, H2:2.0, O:1e-3, OH:1e-3, H2O:1e-1, O2:1e-3, HO2:1e-3, H2O2:1e-3, "
                       "N2:3.76");
  x[0] = gas.temperature();
  gas.getMassFractions(x.data()+1);
}
int main(const int argc, const char *argv[]) {

  const double t1 = strtod(argv[1], NULL);

  IdealGasMix gas("burke_h2_only.xml");
  gas.setState_TPX(300.0, OneAtm,
                   "H:1e-3, H2:2.0, O:1e-3, OH:1e-3, H2O:1e-1, O2:1e-3, HO2:1e-3, H2O2:1e-3, "
                       "N2:3.76");
  RxnFunctor fun(gas);

  std::cout << "Cantera 0D" << std::endl;

  RxnFunctor::state_type x_ref;
  initX(x_ref, gas);
  typedef runge_kutta_cash_karp54<RxnFunctor::state_type> error_stepper_type;
  integrate_adaptive(make_controlled<error_stepper_type>(1.0e-12, 1.0e-12),
                     fun, x_ref, 0.0, t1, 1e-8 );
  cout << "T = " << x_ref[0] << endl;

  // initialize solution vectors
  RxnFunctor::state_type x_t1;
  VectorXd x_t2;

  fun.checkBound(false);
  // creat solver: runge_kutta4_classic
  runge_kutta4_classic< RxnFunctor::state_type > stepper_rk4;
  // create solver: ROWPlus::rosenbrock4
  ROWPlus::rosenbrock4<RxnFunctor, double> stepper_grk4t;
  stepper_grk4t.makeConstantStepper(&fun);
  // creat solver: rosenbrock_krylov4 ROK4A
  rosenbrock_krylov4<RxnFunctor, double> stepper_rok4a(8);
  ODEOptions<double> _opts = stepper_rok4a.getOptions();
  _opts.TypeScheme = ROK4A;
  stepper_rok4a.makeConstantStepper(&fun);
  // creat solver: rosenbrock_krylov4 ROK4E
  rosenbrock_krylov4<RxnFunctor, double> stepper_rok4e(8);
  stepper_rok4e.makeConstantStepper(&fun);

  //
  double dt = 1e-4;
  while (dt >= 1e-7) {
    ROWPlusSolverSpace::Status ret;
    cout << dt << " ";
    // stepper_rk54
    initX(x_t1, gas);
    integrate_const( stepper_rk4 , fun , x_t1 , 0.0 , t1 , dt );
    Map<VectorXd>(x_t1.data(), x_t1.size()) -= Map<VectorXd>(x_ref.data(), x_ref.size());
    cout << Map<VectorXd>(x_t1.data(), x_t1.size()).stableNorm() /
        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
    // stepper_grk4t
    initX(x_t2, gas);
    ret = stepper_grk4t.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
    // stepper_rok4a
    initX(x_t2, gas);
    ret = stepper_rok4a.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
    // stepper_rok4e
    initX(x_t2, gas);
    ret = stepper_rok4e.step(x_t2, 0.0, t1, dt);
    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
      cout << "STAT = " << ret << endl;
      return 1;
    }
    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
    cout << x_t2.stableNorm() /
        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";

    cout << endl;
    dt /= 2.0;
  }
  return 0;
}