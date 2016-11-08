#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H

#include "ODEJac.h"
#include "ODEScheme.h"
#include "ODEUtility.h"
#include <Eigen/Core>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <unistd.h>

// TODO remove after debugging
#include <iostream>

/**
 References:

 [1]  B.A. Schmitt and R. Weiner:
         Matrix-free W-methods using a multiple Arnoldi Iteration,
         APNUM 18(1995), 307-320

 [2]  R. Weiner, B.A. Schmitt an H. Podhaisky:
         ROWMAP - a ROW-code with Krylov techniques for large stiff
         ODEs. Report 39, FB Mathematik und Informatik,
         Universitaet Halle, 1996

 [3]  R.Weiner and B.A.Schmitt:
         Consistency of Krylov-W-Methods in initial value problems,
         Tech. Report 14, FB Mathematik und Informatik,
         Universitaet Halle, 1995

 [4]  E. Hairer and G. Wanner:
         Solving Ordinary Differential Equations II, Springer-Verlag,
         Second Edition, 1996
 */

namespace {

template<typename FunctorType, typename Scalar>
Eigen::DenseIndex fdft1(FunctorType &Functor,
                        Scalar t,
                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
                        &x,
                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &fvec,
                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ft,
                        Scalar epsfcn) {
  using std::sqrt;
  using std::abs;
  using namespace Eigen;

  /* Local variables */
  Scalar h;
  Scalar eps;
  int iflag;

  /* Function Body */
  const Scalar epsmch = NumTraits<Scalar>::epsilon();
  const int n = x.size();
  eigen_assert(fvec.size() == n);

  eps = sqrt((std::max)(epsfcn, epsmch));
  h = eps * abs(t);
  if (h == 0.)
    h = eps;
  iflag = Functor.f(t + h, x, ft);
  if (iflag < 0)
    return iflag;
  ft = ft - fvec;
  ft.noalias() = ft / h;

  return 1;
}
}

namespace ROWPlus {
using namespace Eigen;
using namespace std;

template<typename JacType, typename FunctorType, typename Scalar = double>
class ODESolver {
 public:
  typedef DenseIndex Index;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  ODESolver(FunctorType *_functor,
            ODEOptions <Scalar> &_opt = ODEOptions<Scalar>())
      : opt(_opt), functor(_functor),
        scheme(ODESchemeFactory<Scalar>::make_ODEScheme(opt.TypeScheme)),
        jac(_functor, _opt),
        record(unique_ptr<ODERecord < Scalar> > (new ODERecord<Scalar>())) {
    resizeWork();
  }

  ROWPlusSolverSpace::Status step(VectorType &u, Scalar tstart,
                                  const Scalar tend);

  const ODEStat <Scalar> &getStats() const { return stat; }

  void setInitStepSize(const Scalar step) { opt.h_init = step; }

  void writeSol(const string &fileName) { record->writeRec(fileName); }

  void printStat() { stat.print(); }

 private:
  ODEOptions <Scalar> &opt;
  FunctorType *functor;
  unique_ptr<ODEScheme < Scalar> >
  scheme;
  JacType jac;
  ODEStat <Scalar> stat;
  unique_ptr<ODERecord < Scalar> >
  record;
  Index neq;
  VectorType fm, fdt, scal, rhs, uu, fu0, wa;
  MatrixType km;

  void resizeWork();
};

template<typename JacType, typename FunctorType, typename Scalar>
ROWPlusSolverSpace::Status
ODESolver<JacType, FunctorType, Scalar>::step(VectorType &u, Scalar t,
                                              const Scalar tend) {
  // verify the size of input solution vector
  eigen_assert(u.size() == neq);
  // clear stats
  stat.clear();
  jac.initJacReuse();
  // update options for Jac
  jac.updateOptions(opt);
  // local variables
  const Scalar sqrt_neq = sqrt((Scalar) neq);
  bool reached = false, rejected = false, failed = false, first = true;
  Scalar unorm = 0.0;
  Scalar hs, hnew, ts, told;
  Scalar ehg, errs = 1.0, errold = 1.0;
  Index nret;

  // set-up itial step size if not specified
  if (opt.h_init == 0.0) {
    unorm = u.blueNorm() / sqrt_neq;
    hs = unorm * opt.relTol + opt.absTol;
    hs = scheme->proot(1.0 / hs) * 1.0e-1;
  } else {
    hs = opt.h_init;
  }
  hs = min(hs, opt.h_max);
  // evaluate rhs at the beginning
  if (functor->f(t, u, fm) < 0)
    return ROWPlusSolverSpace::UserAsked;
  stat.nfeval++;

  // Begin integration, loop for successful steps:
  while (true) {
    // handle previous failure first
    if (failed) {
      stat.nstepsr++;
      rejected = true;
      reached = false;
      hs *= opt.stepControl[0];
      // reset failure flag
      failed = false;
    }
    // reached
    if (t >= tend || reached)
      break;
    // too many steps
    if (stat.nsteps > opt.maxSteps)
      return ROWPlusSolverSpace::TooManySteps;
    // Adjust step size if close to tend
    if ((t + hs) >= tend) {
      hs = tend - t;
      reached = true;
    }
    scal = u.array() * opt.relTol + opt.absTol;
    scal.noalias() = scal.cwiseInverse();
    // TODO remove after dev
    // cout << "hs = " << hs << ", t = " << t << "/" << tend << endl;
    // cout << "=====u=====" << endl << u << endl << "===" << endl;
    // cout << "=====scal=====" << endl << scal << endl << "===" << endl;
    // Evaluate f(u0)
    if (!rejected && scheme->cf[0] && !first) {
      if (functor->f(t, u, fm) < 0) {
        if (opt.iUserAskedKill)
          return ROWPlusSolverSpace::UserAsked;
        else {
          failed = true;
          continue;
        }
      }
      stat.nfeval++;
    }
    // initialize ehg*I-J
    ehg = 1.0 / (scheme->gamma * hs);
    nret = jac.init(t, u, fm, ehg, rejected);
    if (nret < 0) {
      if (opt.iUserAskedKill)
        return ROWPlusSolverSpace::UserAsked;
      else {
        failed = true;
        continue;
      }
    }
    if (opt.iUserJac)
      stat.njac += nret;
    else
      stat.nfeval += nret;
    // Compute the derivative f_t in the nonautonomous case.
    if (!opt.iAuto) {
      if (opt.iUserFt) {
        functor->fdt(t, u, fdt);
        stat.nfdt++;
      } else {
        if (fdft1(*functor, t, u, fm, fdt, opt.epsfcn) < 0) {
          if (opt.iUserAskedKill)
            return ROWPlusSolverSpace::UserAsked;
          else {
            failed = true;
            continue;
          }
        }
        stat.nfeval += 1;
      }
    }
    // Loop over stages
    // 1st stage
    rhs = fm;
    if (jac.type() != ZRO) {
      rhs *= ehg;
      if (!opt.iAuto)
        rhs += (ehg * hs * scheme->di(0)) * fdt;
      jac.stage(0, rhs, rhs);
    }
    km.col(0) = rhs;
    // rest stages
    for (size_t i = 1; i < scheme->nStage; i++) {
      // f(u0 + sum_j^{i-1} a_{ij} k_j)
      // last stage may be omitted if e.g. a41=a31, a42=a32, a43=0
      if (scheme->cf[i]) {
        uu = u;
        for (size_t j = 0; j < i; j++) {
          uu += (hs * scheme->aij(i - 1, j)) * km.col(j);
        }
        ts = t + scheme->ci(i - 1) * hs;
        if (functor->f(ts, uu, rhs) < 0) {
          if (opt.iUserAskedKill)
            return ROWPlusSolverSpace::UserAsked;
          else {
            failed = true;
            break;
          }
        }
        stat.nfeval++;
        // if the next stage is omitted store rhs to km.col(next)
        if (i + 1 < scheme->nStage && !scheme->cf[i + 1]) {
          km.col(i + 1) = rhs;
        }
        // if FSAL stor rhs in fm for last stage
        if (!scheme->cf[0] && i == scheme->nStage - 1)
          fm = rhs;
      } else {
        rhs = km.col(i);
      }
      // rhs += h*d_i*f_t
      if (!opt.iAuto)
        rhs += (hs * scheme->di(i)) * fdt;
      if (jac.type() != ZRO) {
        // rhs = f(u0 + sum_j^{i-1} a_{ij} k_j) + sum_j^{i-1} c_{ij} k_j
        for (size_t j = 0; j < i; j++) {
          rhs += scheme->cij(i - 1, j) * km.col(j);
        }
        rhs *= ehg;
        // rhs = (I-h /gamma_i T)^{-1} rhs
        jac.stage(i, rhs, wa);
        // k(i) = rhs - sum_j^{i-1} c_{ij} k_j
        for (size_t j = 0; j < i; j++) {
          wa -= scheme->cij(i - 1, j) * km.col(j);
        }
        km.col(i) = wa;
      } else {
        km.col(i) = rhs;
      }
    }
    // if failed skip the rest and continue the while loop;
    if (failed)
      continue;
    // New solution: uu = u + sum (h* b.i * ak.i,i=1..4).
    uu = u;
    for (size_t i = 0; i < scheme->nStage; i++) {
      uu += (hs * scheme->bi(i)) * km.col(i);
    }
    // Embedded solution: fu0 = sum (hs* e.i * ak.i, i=1..4).
    fu0.setZero();
    for (size_t i = 0; i < scheme->nStage; i++) {
      fu0 += (hs * scheme->ei(i)) * km.col(i);
    }

    // Error estimate, step size control.
    errs = fu0.cwiseProduct(scal).stableNorm();
    errs /= sqrt_neq;
    hnew = min(hs * min(opt.stepControl[1],
                        max(opt.stepControl[0], scheme->proot(1.0 / errs) *
                            scheme->pproot(errold) *
                            opt.stepControl[2])),
               opt.h_max);
    if (hnew < opt.h_min) {
      return ROWPlusSolverSpace::StepSizeTooSmall;
    }
    // Check positivity
    if (opt.iCheckPos && !functor->checkPositivity(uu)) {
      hnew = opt.stepControl[0] * hs;
      errs = 2.;
    }
    if (errs < 1.0) { // Step is accepted.
      if (opt.iWrite)
        record->addSol(t, uu);
      told = t;
      t += hs;
      u = uu;
      stat.nsteps++;
      stat.lasths = hs;
      rejected = false;
      first = false;
      hs = hnew;
      errold = errs;
    } else { // Step is rejected.
      stat.nstepsr++;
      rejected = true;
      reached = false;
      hs = hnew;
      // jac.initJacReuse();
    }
  }

  return ROWPlusSolverSpace::ComputeSucessful;
}

template<typename JacType, typename FunctorType, typename Scalar>
void ODESolver<JacType, FunctorType, Scalar>::resizeWork() {
  neq = functor->inputs();
  eigen_assert(neq == functor->values());

  // resize everything
  fm.resize(neq);
  scal.resize(neq);
  rhs.resize(neq);
  uu.resize(neq);
  fu0.resize(neq);
  wa.resize(neq);
  if (!opt.iAuto)
    fdt.resize(neq);
  km.resize(neq, scheme->nStage);
}
}
#endif
