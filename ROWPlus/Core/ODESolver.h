#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H

#include <memory>
#include <stdexcept>

#include <string>
#include <iostream>

#include <Eigen/Core>

#include "ROWPlus/Core/ODEJac.h"
#include "ROWPlus/Core/ODEScheme.h"
#include "ROWPlus/Core/ODEUtility.h"

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

namespace ROWPlus {

template<typename JacType, typename FunctorType, typename Scalar = double>
class ODESolver {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODESolver(FunctorType *_functor,
            ODEOptions<Scalar> &_opt = ODEOptions<Scalar>())
      : opt(_opt), functor(_functor),
        scheme(ODESchemeFactory<Scalar>::make_ODEScheme(opt.TypeScheme)),
        jac(_functor, _opt) {
    resizeWA();
  }

  ROWPlusSolverSpace::Status step(Eigen::Ref<VectorType> u,
                                  Scalar start_time,
                                  const Scalar end_time);

  const ODEStat<Scalar> &getStats() const { return stat; }

  void setInitStepSize(const Scalar step) { opt.h_init = step; }

 private:
  FunctorType *functor;
  ODEOptions<Scalar> &opt;
  std::unique_ptr<ODEScheme<Scalar>> scheme;
  JacType jac;
  ODEStat<Scalar> stat;
  Index neq;
  VectorType fm, fdt, scal, rhs, uu, fu0, wa;
  MatrixType km;

  int dfdt(FunctorType &Functor,
           Scalar t,
           const Eigen::Ref<const VectorType> x,
           const Eigen::Ref<const VectorType> fvec,
           Eigen::Ref<VectorType> ft, const Scalar epsfcn);
  inline int evalFt(Scalar t,
                    const Eigen::Ref<const VectorType> u,
                    const Eigen::Ref<const VectorType> f,
                    Eigen::Ref<VectorType> ft);
  inline int evalF(Scalar t, const Eigen::Ref<const VectorType> u,
                   Eigen::Ref<VectorType> f);
  void resizeWA();
};

template<typename JacType, typename FunctorType, typename Scalar>
ROWPlusSolverSpace::Status
ODESolver<JacType, FunctorType, Scalar>::step(Eigen::Ref<VectorType> u,
                                              Scalar t, const Scalar end_time) {
  // verify the size of input solution vector
  eigen_assert(u.size() == neq);
  // clear stats
  stat.clear();
  // update options for Jac
  jac.updateOptions(opt);
  // local variables
  const Scalar sqrt_neq = sqrt((Scalar) neq);
  bool reached = false, rejected = false, failed = false, first = true;
  Scalar unorm = 0.0;
  Scalar hs, hnew, ts, told;
  Scalar ehg, errs = 1.0, errold = 1.0;
  Index nret;

  // set-up initial step size if not specified
  if (opt.h_init == 0.0) {
    unorm = u.blueNorm() / sqrt_neq;
    hs = unorm * opt.relTol + opt.absTol;
    hs = scheme->proot(1.0 / hs) * 1.0e-1;
  } else {
    hs = opt.h_init;
  }
  hs = std::min(hs, opt.h_max);
  // evaluate rhs at the beginning
  if (evalF(t, u, fm) < 0) return ROWPlusSolverSpace::UserAsked;

  // begin integration, loop for successful steps:
  while (true) {
    // handle previous failure first
    if (failed) {
      stat.nstepsr++;
      rejected = true;
      reached = false;
      hs *= opt.stepControl[0];
      failed = false;
    }
    // reached
    if (t >= end_time || reached) break;
    // too many steps
    if (stat.nsteps > opt.maxSteps) return ROWPlusSolverSpace::TooManySteps;
    // adjust step size if close to end_time
    if ((reached = (t + hs >= end_time))) hs = end_time - t;
    scal = u.array() * opt.relTol + opt.absTol;
    scal.noalias() = scal.cwiseInverse();
    // evaluate f(u0)
    if (!rejected && !first && scheme->cf[0])
      if ((failed = (evalF(t, u, fm) < 0))) {
        if (opt.iUserAskedKill) return ROWPlusSolverSpace::UserAsked;
        else continue;
      }
    // initialize ehg*I-J
    ehg = 1.0 / (scheme->gamma * hs);
    nret = jac.init(t, u, fm, ehg, rejected);
    if ((failed = (nret < 0))) {
      if (opt.iUserAskedKill) return ROWPlusSolverSpace::UserAsked;
      else continue;
    }
    if (opt.iUserJac) stat.njac += nret;
    else stat.nfeval += nret;
    // Compute the derivative f_t in the nonautonomous case.
    if (!opt.iAuto)
      if ((failed = (evalFt(t, u, fm, fdt) < 0))) {
        if (opt.iUserAskedKill) return ROWPlusSolverSpace::UserAsked;
        else continue;
      }
    // Loop over stages
    // 1st stage
    rhs = fm;
    if (jac.type() != ZRO) {
      rhs *= ehg;
      if (!opt.iAuto) rhs += (ehg * hs * scheme->di(0)) * fdt;
      jac.stage(0, rhs, rhs);
    }
    km.col(0) = rhs;
    // rest stages
    for (size_t i = 1; i < scheme->nStage; ++i) {
      // f(u0 + sum_j^{i-1} a_{ij} k_j)
      // last stage may be omitted if e.g. a41=a31, a42=a32, a43=0
      if (scheme->cf[i]) {
        uu = u;
        for (size_t j = 0; j < i; ++j)
          uu += (hs * scheme->aij(i - 1, j)) * km.col(j);
        ts = t + scheme->ci(i - 1) * hs;
        if ((failed = (evalF(ts, uu, rhs) < 0))) {
          if (opt.iUserAskedKill) return ROWPlusSolverSpace::UserAsked;
          else break;
        }
        // if the next stage is omitted store rhs to km.col(next)
        if (i + 1 < scheme->nStage && !scheme->cf[i + 1]) km.col(i + 1) = rhs;
        // if FSAL stor rhs in fm for last stage
        if (!scheme->cf[0] && i == scheme->nStage - 1) fm = rhs;
      } else rhs = km.col(i);
      // rhs += h*d_i*f_t
      if (!opt.iAuto) rhs += (hs * scheme->di(i)) * fdt;
      if (jac.type() != ZRO) {
        // rhs = f(u0 + sum_j^{i-1} a_{ij} k_j) + sum_j^{i-1} c_{ij} k_j
        for (size_t j = 0; j < i; j++) rhs += scheme->cij(i - 1, j) * km.col(j);
        rhs *= ehg;
        // rhs = (I-h /gamma_i T)^{-1} rhs
        jac.stage(i, rhs, wa);
        // k(i) = rhs - sum_j^{i-1} c_{ij} k_j
        for (size_t j = 0; j < i; j++) wa -= scheme->cij(i - 1, j) * km.col(j);
        km.col(i) = wa;
      } else km.col(i) = rhs;
    }
    // if failed skip the rest and continue the while loop;
    if (failed) continue;
    // New solution: uu = u + sum (h* b.i * ak.i,i=1..4).
    uu = u;
    for (size_t i = 0; i < scheme->nStage; i++)
      uu += (hs * scheme->bi(i)) * km.col(i);
    // Embedded solution: fu0 = sum (hs* e.i * ak.i, i=1..4).
    fu0.setZero();
    for (size_t i = 0; i < scheme->nStage; i++)
      fu0 += (hs * scheme->ei(i)) * km.col(i);
    // Error estimate, step size control.
    errs = fu0.cwiseProduct(scal).stableNorm();
    errs /= sqrt_neq;
    hnew = std::min(hs * std::min(opt.stepControl[1],
                                  std::max(opt.stepControl[0],
                                           scheme->proot(1.0 / errs) *
                                               scheme->pproot(errold) *
                                               opt.stepControl[2])),
                    opt.h_max);
    if (hnew < opt.h_min && end_time > opt.h_min) {
      return ROWPlusSolverSpace::StepSizeTooSmall;
    }
    if (errs < 1.0) { // Step is accepted.
//      std::cout << t << " " << hs << std::endl;
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
int ODESolver<JacType, FunctorType, Scalar>::dfdt(FunctorType &Functor,
                                                  Scalar t,
                                                  const Eigen::Ref<const VectorType> x,
                                                  const Eigen::Ref<const VectorType> fvec,
                                                  Eigen::Ref<VectorType> ft,
                                                  const Scalar epsfcn) {
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

template<typename JacType, typename FunctorType, typename Scalar>
int ODESolver<JacType, FunctorType, Scalar>::evalFt(Scalar t,
                                                    const Eigen::Ref<const VectorType> u,
                                                    const Eigen::Ref<const VectorType> f,
                                                    Eigen::Ref<VectorType> ft) {
  if (opt.iUserFt) {
    stat.nfdt++;
    return functor->fdt(t, u, ft);
  } else {
    stat.nfeval += 1;
    return dfdt(*functor, t, u, fm, fdt, opt.epsfcn);
  }
};

template<typename JacType, typename FunctorType, typename Scalar>
int ODESolver<JacType, FunctorType, Scalar>::evalF
    (Scalar t, const Eigen::Ref<const VectorType> u, Eigen::Ref<VectorType> f) {
  stat.nfeval++;
  return functor->f(t, u, f);
};

template<typename JacType, typename FunctorType, typename Scalar>
void ODESolver<JacType, FunctorType, Scalar>::resizeWA() {
  neq = functor->values();
  eigen_assert(neq == functor->values());

  // resize everything
  fm.resize(neq);
  scal.resize(neq);
  rhs.resize(neq);
  uu.resize(neq);
  fu0.resize(neq);
  wa.resize(neq);
  if (!opt.iAuto) fdt.resize(neq);
  km.resize(neq, scheme->nStage);
}

}
#endif
