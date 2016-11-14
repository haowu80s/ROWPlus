//
// Created by Hao Wu on 11/11/16.
//

#include <iostream>
#include <boost/numeric/odeint.hpp>
#include "ROWPlus/ROWPlus.h"
#include "RxnFunctor.h"

/* Header files for CVODE */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */


using namespace Eigen;
using namespace ROWPlus;
using namespace boost::numeric::odeint;
using namespace Cantera;
using namespace std;

template<typename vec>
static void initX(vec& x, IdealGasMix& gas) {
  x.resize(gas.nSpecies()+1);
  gas.setState_TPX(1500.0, OneAtm, "CH4:0.5, O2:1.0, N2:3.76, AR:1e-6");
  x[0] = gas.temperature();
  gas.getMassFractions(x.data()+1);
}

static void initXCvode(N_Vector x, IdealGasMix& gas) {
  gas.setState_TPX(1500.0, OneAtm, "CH4:0.5, O2:1.0, N2:3.76, AR:1e-6");
  NV_Ith_S(x,0) = gas.temperature();
  gas.getMassFractions(NV_DATA_S(x) + 1);
}

static int check_flag(void *flagvalue, const string& funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname.c_str());
    return(1); }

    /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname.c_str(), *errflag);
      return(1); }}

    /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname.c_str());
    return(1); }

  return(0);
}

extern "C"
{
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
  double *ydata = NV_DATA_S(y);
  double *dydata = NV_DATA_S(ydot);
  RxnFunctor *fun = (RxnFunctor *) user_data;
  return fun->f(t, Map<VectorXd>(ydata, fun->inputs()),
                Map<VectorXd>(dydata, fun->inputs()));
}
}

int main(const int argc, const char *argv[]) {

  double TIME1 = strtod(argv[1], NULL);
  double DTIME = 1e-4;

  IdealGasMix gas("gri30.xml");
  RxnFunctor fun(gas);
  fun.checkBound(false);
  std::cout << "Cantera 0D" << std::endl;

  /* Create serial vector of length NEQ for I.C.*/
  size_t NEQ = fun.inputs();
  realtype cvode_t1, cvode_hlast, cvode_t;
  N_Vector cvode_y = N_VNew_Serial(NEQ);
  void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  /* Initialize y */
  initXCvode(cvode_y, gas);
  /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula and the use of a Newton iteration */
  int flag = CVodeInit(cvode_mem, f, 0.0, cvode_y);
  flag = CVodeSStolerances(cvode_mem, 1e-12, 1e-12);
  flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
  flag = CVodeSetUserData(cvode_mem, &fun);
  flag = CVDense(cvode_mem, NEQ);

  // Generate reference solution
  CVode(cvode_mem, cvode_t1 = TIME1, cvode_y, &cvode_t, CV_NORMAL);
  assert(cvode_t == cvode_t1);
  cout << "T_end = " << NV_Ith_S(cvode_y,0) << endl;
  VectorXd y_ref = Map<const VectorXd>(NV_DATA_S(cvode_y), fun.inputs());

  // CVODE solution
  initXCvode(cvode_y, gas);
  flag = CVodeReInit(cvode_mem, 0.0, cvode_y);
  flag = CVodeSStolerances(cvode_mem, 1e-8, 1e-10);
  CVode(cvode_mem, cvode_t1 = TIME1, cvode_y, &cvode_t, CV_NORMAL);
  assert(cvode_t == cvode_t1);
  cout << "T_end = " << NV_Ith_S(cvode_y,0) << endl;
  VectorXd y_cvode = Map<const VectorXd>(NV_DATA_S(cvode_y), fun.inputs());
  y_cvode -= y_ref;
  y_cvode.array() /= (y_ref.array() + 1e-6);
  cout << y_cvode.stableNorm() / sqrt(y_cvode.size()) << endl;


  VectorXd rowplus_y;
  double rowplus_hlast, rowplus_t;
  rosenbrock_krylov4<RxnFunctor, double> stepper_rok4e(4);
  initX(rowplus_y, gas);
  stepper_rok4e.makeAdaptiveStepper(&fun, 1e-6, 1e-8);
  ROWPlusSolverSpace::Status ret = stepper_rok4e.step(rowplus_y, 0.0, TIME1,
                                                      1e-9);
  if (ret != ROWPlusSolverSpace::ComputeSucessful) {
    cout << "STAT = " << ret << endl;
    return 1;
  }
  cout << "T_end = " << rowplus_y[0] << " " << rowplus_y.minCoeff() << endl;
  rowplus_y -= y_ref;
  rowplus_y.array() /= (y_ref.array() + 1e-6);
  cout << rowplus_y.stableNorm() / sqrt(rowplus_y.size()) << endl;
  /* Free y and abstol vectors */
  N_VDestroy_Serial(cvode_y);
  /* Free integrator memory */
  CVodeFree(&cvode_mem);
  return 0;
};
//  // initialize solution vectors
//  RxnFunctor::state_type x_t1;
//  VectorXd x_t2;
//
//  fun.checkBound(false);
//  // creat solver: runge_kutta4_classic
//  runge_kutta4_classic< RxnFunctor::state_type > stepper_rk4;
//  // create solver: ROWPlus::rosenbrock4
//  ROWPlus::rosenbrock4<RxnFunctor, double> stepper_grk4t;
//  stepper_grk4t.makeConstantStepper(&fun);
//  // creat solver: rosenbrock_krylov4 ROK4A
//  rosenbrock_krylov4<RxnFunctor, double> stepper_rok4a(8);
//  auto _opts = stepper_rok4a.getOptions();
//  _opts.TypeScheme = ROK4A;
//  stepper_rok4a.setOptions(_opts);
//  stepper_rok4a.makeConstantStepper(&fun);
//  // creat solver: rosenbrock_krylov4 ROK4E
//  rosenbrock_krylov4<RxnFunctor, double> stepper_rok4e(8);
//  stepper_rok4e.makeConstantStepper(&fun);
//
//  //
//  double dt = 1e-4;
//  while (dt >= 1e-7) {
//    ROWPlusSolverSpace::Status ret;
//    cout << dt << " ";
//    // stepper_rk54
//    initX(x_t1, gas);
//    integrate_const( stepper_rk4 , fun , x_t1 , 0.0 , t1 , dt );
//    Map<VectorXd>(x_t1.data(), x_t1.size()) -= Map<VectorXd>(x_ref.data(), x_ref.size());
//    cout << Map<VectorXd>(x_t1.data(), x_t1.size()).stableNorm() /
//        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
//    // stepper_grk4t
//    initX(x_t2, gas);
//    ret = stepper_grk4t.step(x_t2, 0.0, t1, dt);
//    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
//      cout << "STAT = " << ret << endl;
//      return 1;
//    }
//    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
//    cout << x_t2.stableNorm() /
//        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
//    // stepper_rok4a
//    initX(x_t2, gas);
//    ret = stepper_rok4a.step(x_t2, 0.0, t1, dt);
//    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
//      cout << "STAT = " << ret << endl;
//      return 1;
//    }
//    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
//    cout << x_t2.stableNorm() /
//        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
//    // stepper_rok4e
//    initX(x_t2, gas);
//    ret = stepper_rok4e.step(x_t2, 0.0, t1, dt);
//    if (ret != ROWPlusSolverSpace::ComputeSucessful) {
//      cout << "STAT = " << ret << endl;
//      return 1;
//    }
//    x_t2 -= Map<VectorXd>(x_ref.data(), x_ref.size());
//    cout << x_t2.stableNorm() /
//        Map<VectorXd>(x_ref.data(), x_ref.size()).stableNorm() << " ";
//
//    cout << endl;
//    dt /= 2.0;
//  }
//}