#ifndef ODE_UTILITY_H
#define ODE_UTILITY_H

#include <Eigen/Core>
#include <fstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

// TODO remove after debugging
#include <iostream>

namespace ROWPlus {
using namespace Eigen;
using namespace std;

// Generic functor
// for the evaluation of f(t, u), f_u(t, u), f_t(t, U), etc.
template <typename _Scalar, DenseIndex NX = Dynamic, DenseIndex NY = Dynamic>
struct BaseFunctor {
  typedef _Scalar Scalar;
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  BaseFunctor() {}

  DenseIndex inputs() const { return InputsAtCompileTime; }
  DenseIndex values() const { return ValuesAtCompileTime; }

  // you should define that in the subclass :
  // void operator() (const InputType& x, ValueType* v);
  // void f() (Scalar t, const InputType& x, ValueType* v);
  // void df() (Scalar t, const InputType& x, ValueType* v);
};

/**
 * Types of Rosenbrock schemes
 */
enum ODESchemeType {
  SHAMP = 0, // (Shampine (1982))    see [4], page 110
  GRK4T = 1, // (Kaps-Rentrop 79)    see [4], page 110
  VELDS = 2, // (Veldhuizen (1984))  see [4], page 110
  VELDD = 3, // (Veldhuizen (1984))  see [4], page 110
  LSTAB = 4, // (L-stable method)    see [4], page 110
  GRK4A = 5, // (Kaps-Rentrop 79)    see [4], page 110
  ROK4A = 6, // (Rosenbrock-Krylov, Tranquilli and Sandu (2014))
  ROK4L = 7, // (Improved Rosenbrock-Krylov)
  RKDP = 8,  // (Dormand & Prince 1980)
  RK23 = 9,  // (Bogacki–Shampine 1989, same as ode23 in MATLAB)
  ZED34 = 10 // (H. Zedan 1989)
};

/**
 * Types of low rank approx. to the Jacobian matrix
 */
enum ODEJacType {
  ZRO = 0, // Zero matrix (=== explict method)
  EXA = 1, // Exact Jacobian (=== standard Rosenbrock)
  SAP = 2, // Single Arnoldi process
  RAP = 3  // Jacobian-Reuse Arnoldi process
};

template <typename Scalar = double> struct ODEOptions {
  typedef DenseIndex Index;
  ODEOptions()
      : TypeScheme(GRK4T), relTol(1e-4),
        absTol(sqrt(NumTraits<Scalar>::epsilon())), h_init(0.0),
        h_min(NumTraits<Scalar>::epsilon()),
        h_max(NumTraits<Scalar>::highest()), maxSteps(100000), iAuto(true),
        iUserJac(false), iUserFt(false), iUserAskedKill(false), iWrite(false),
        stepControl{0.25, 4.0, 0.8}, epsfcn(Scalar(0.)),
        epsmch(NumTraits<Scalar>::epsilon()), kryTol(1e-1), maxKryDim(4),
        minKryDim(4), maxJacReuse(0), iCheckPos(false) {}
  ODESchemeType TypeScheme;
  Scalar relTol;
  Scalar absTol;
  Scalar h_init;
  Scalar h_min;
  Scalar h_max;
  Index maxSteps;
  bool iAuto;
  bool iUserJac;
  bool iUserFt;
  bool iUserAskedKill;
  bool iWrite;
  Scalar stepControl[3]; // Parameters for step size selection.
  // eps for Scalar
  Scalar epsfcn;
  Scalar epsmch;
  // options for krylov type
  Scalar kryTol;
  Index maxKryDim;
  Index minKryDim;
  // options for Jacobian reuse
  Index maxJacReuse;
  // check positivity
  bool iCheckPos;
};

template <typename Scalar = double> struct ODEStat {
  ODEStat()
      : nsteps(0), nstepsr(0), nfeval(0), nfdt(0), njac(0), njacv(0),
        lasths(-1.0) {}

  int nsteps,    // Number of computed (accepted and rejected) steps.
      nstepsr,   // Number of rejected steps.
      nfeval,    // Number of function evaluations.
      nfdt,      // Number of dF(t, u)/dt evaluations.
      njac,      // Number of Jacobian matrix evaludations.
      njacv;     // Number of Jacobian-times-vector products.
  Scalar lasths; // Latest time step size.
  void print() const {
    cout << "====================" << endl;
    cout << "|| nsteps  = " << nsteps << endl;
    cout << "|| nstepsr = " << nstepsr << endl;
    cout << "|| nfeval  = " << nfeval << endl;
    cout << "|| nfdt    = " << nfdt << endl;
    cout << "|| njac    = " << njac << endl;
    cout << "|| njacv   = " << njacv << endl;
    cout << "|| lasths  = " << njacv << endl;
    cout << "====================" << endl;
  }
  void clear() {
    nsteps = 0;
    nstepsr = 0;
    nfeval = 0;
    nfdt = 0;
    njac = 0;
    njacv = 0;
    lasths = -1.0;
  }
};

template <typename Scalar = double> class ODERecord {
public:
  typedef DenseIndex Index;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  void addSol(const Scalar t, const VectorType &u) {
    vector<Scalar> uVec(u.data(), u.data() + u.size());
    tRecords.push_back(t);
    uRec.push_back(uVec);
  }

  void writeRec(const string &fileName) {
    Index nRec = tRecords.size();
    ofstream file(fileName);
    if (file.is_open()) {
      for (size_t i = 0; i < nRec; i++) {
        file << tRecords[i] << " ";
        for (auto it = uRec[i].begin(); it != uRec[i].end(); it++) {
          file << *it << " ";
        }
        file << "\n";
      }
    }
    file.close();
  }

private:
  vector<Scalar> tRecords;
  vector<vector<Scalar> > uRec;
};

namespace ROWPlusSolverSpace {

enum Status {
  ComputeSucessful = 1,
  ComputeInterrupted = 2,
  StepSizeTooSmall = -1,
  TooManySteps = -2,
  ImproperInputParameters = -3,
  JacMatSingular = -4,
  UserAsked = -5
};
}
}

#endif
