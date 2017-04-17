#ifndef ODE_JAC_H
#define ODE_JAC_H

#include "ROWPlus/Core/ODEUtility.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdexcept>
#include <stdio.h>

namespace ROWPlus {

/**
 * CRTP for static polymorphism
 */
template<class T, typename FunctorType, typename Scalar = double>
class ODEJac {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODEJac(FunctorType *_functor, ODEJacType _tyepName,
         const ODEOptions <Scalar> &_opt)
      : functor(_functor),
        typeName(_tyepName),
        neq(functor->values()),
        iUserJac(_opt.iUserJac),
        maxKryDim(_opt.maxKryDim),
        epsfcn(_opt.epsfcn),
        arnTol(_opt.absTol),
        iReSize(true) {}

  ODEJacType type() const { return typeName; }

  Index init(Scalar t,
             Eigen::Ref<VectorType> u,
             Eigen::Ref<VectorType> f,
             Scalar _ehg,
             bool rejected = false) {
    return static_cast<T *>(this)->initIMPL(t, u, f, _ehg, rejected);
  };

  Index stage(Index s,
              const Eigen::Ref<const VectorType> b,
              Eigen::Ref<VectorType> x) {
    return static_cast<T *>(this)->stageIMPL(s, b, x);
  };

  void updateOptions(const ODEOptions <Scalar> &_opt) {
    iReSize = (maxKryDim != _opt.maxKryDim);
    iUserJac = _opt.iUserJac;
    maxKryDim = _opt.maxKryDim;
    epsfcn = _opt.epsfcn;
    arnTol = _opt.absTol;
  }

 protected:
  FunctorType *functor;
  const ODEJacType typeName;
  bool iUserJac;
  bool iReSize;
  Index neq;
  Index maxKryDim;
  Scalar epsfcn;
  Scalar arnTol;

  Index fdjac(FunctorType &Functor, Scalar t,
              Eigen::Ref<VectorType> x,
              const Eigen::Ref<const VectorType> fvec,
              Eigen::Ref<MatrixType> fjac,
              Index ml,
              Index mu,
              Scalar epsfcn);

  Index fdjacv(FunctorType &Functor,
               Scalar t,
               const Eigen::Ref<const VectorType> x,
               const Eigen::Ref<const VectorType> fvec,
               const Eigen::Ref<const VectorType> vvec,
               Eigen::Ref<VectorType> jacv,
               Scalar epsfcn);
};

template<class T, typename FunctorType, typename Scalar>
Eigen::DenseIndex ODEJac<T, FunctorType, Scalar>::fdjac(FunctorType &Functor,
                                                        Scalar t,
                                                        Eigen::Ref<VectorType> x,
                                                        const Eigen::Ref<const VectorType> fvec,
                                                        Eigen::Ref<MatrixType> fjac,
                                                        Eigen::DenseIndex ml,
                                                        Eigen::DenseIndex mu,
                                                        Scalar epsfcn) {

  using namespace Eigen;
  using std::sqrt;
  using std::abs;

  /* Local variables */
  Scalar h;
  Index j, k;
  Scalar eps, temp;
  Index msum;
  int iflag;
  Index start, length;

  /* Function Body */
  const Scalar epsmch = Eigen::NumTraits<Scalar>::epsilon();
  const Index n = x.size();
  eigen_assert(fvec.size() == n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> wa1(n);
  Matrix<Scalar, Eigen::Dynamic, 1> wb1(n);

  eps = sqrt((std::max)(epsfcn, epsmch));
  msum = ml + mu + 1;
  if (msum >= n) {
    /* computation of dense approximate jacobian. */
    for (j = 0; j < n; ++j) {
      temp = x[j];
      h = eps * abs(temp);
      if (h == 0.)
        h = eps;
      x[j] = temp + h;
      iflag = Functor.f(t, x, wa1);
      if (iflag < 0)
        return iflag;
      x[j] = temp;
      wa1 -= fvec;
      wa1 /= h;
      fjac.col(j) = wa1;
    }
  } else {
    /* computation of banded approximate jacobian. */
    for (k = 0; k < msum; ++k) {
      for (j = k; (msum < 0) ? (j > n) : (j < n); j += msum) {
        wb1[j] = x[j];
        h = eps * abs(wb1[j]);
        if (h == 0.)
          h = eps;
        x[j] = wb1[j] + h;
      }
      iflag = Functor.f(t, x, wa1);
      if (iflag < 0)
        return iflag;
      for (j = k; (msum < 0) ? (j > n) : (j < n); j += msum) {
        x[j] = wb1[j];
        h = eps * abs(wb1[j]);
        if (h == 0.)
          h = eps;
        fjac.col(j).setZero();
        start = std::max<Index>(0, j - mu);
        length = (std::min)(n - 1, j + ml) - start + 1;
        fjac.col(j).segment(start, length) =
            (wa1.segment(start, length) - fvec.segment(start, length)) / h;
      }
    }
  }
  return 0;
}

template<class T, typename FunctorType, typename Scalar>
Eigen::DenseIndex ODEJac<T, FunctorType, Scalar>::fdjacv(FunctorType &Functor,
                                                         Scalar t,
                                                         const Eigen::Ref<const VectorType> x,
                                                         const Eigen::Ref<const VectorType> fvec,
                                                         const Eigen::Ref<const VectorType> vvec,
                                                         Eigen::Ref<VectorType> jacv,
                                                         Scalar epsfcn) {
  using std::sqrt;
  using std::abs;
  using namespace Eigen;

  /* Local variables */
  Scalar eps;
  int iflag;

  /* Function Body */
  const Scalar epsmch = Eigen::NumTraits<Scalar>::epsilon();
  const Index n = x.size();
  eigen_assert(fvec.size() == n);
  eps = sqrt((std::max)(epsfcn, epsmch)) /
        std::max(vvec.blueNorm() * sqrt((Scalar) n), epsmch);
  Matrix<Scalar, Eigen::Dynamic, 1> wa1(n);

  wa1 = x + eps * vvec;
  iflag = Functor.f(t, wa1, jacv);
  if (iflag < 0)
    return iflag;
  jacv -= fvec;
  jacv /= eps;
  return 0;
}

template<typename FunctorType, typename Scalar = double>
class ODEJacZRO
    : public ODEJac<ODEJacZRO<FunctorType, Scalar>, FunctorType, Scalar> {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODEJacZRO(FunctorType *_functor, const ODEOptions <Scalar> &_opt)
      : ODEJac<ODEJacZRO<FunctorType, Scalar>, FunctorType, Scalar>(_functor,
                                                                    ZRO, _opt) {
  }

  Index initIMPL(Scalar t,
                 Eigen::Ref<VectorType> u,
                 Eigen::Ref<VectorType> f,
                 Scalar _ehg,
                 bool rejected = false) {
    return 0;
  }

  Index stageIMPL(Index s,
                  const Eigen::Ref<const VectorType> b,
                  Eigen::Ref<VectorType> x) { return 0; };
};

/**
 * \brief Exact Jacobian matrix for Rosenbrock-type methods
 * Jacobian is evaluated either analysitically or by finite difference.
 * The linear system is solved by partial pivoting LU, which is not
 * rank-revealing.
 * I-ehg*J is assumed to be full-rank without checking.
 */
template<typename FunctorType, typename Scalar = double>
class ODEJacEXA
    : public ODEJac<ODEJacEXA<FunctorType, Scalar>, FunctorType, Scalar> {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODEJacEXA(FunctorType *_functor, const ODEOptions <Scalar> &_opt)
      : ODEJac<ODEJacEXA<FunctorType, Scalar>, FunctorType, Scalar>
            (_functor, EXA, _opt),
        J(MatrixType(this->neq, this->neq)),
        MLU(Eigen::PartialPivLU<MatrixType>(this->neq)) {
    eigen_assert(this->neq == this->functor->values());
  }

  Index initIMPL(Scalar t,
                 Eigen::Ref<VectorType> u,
                 Eigen::Ref<VectorType> f,
                 Scalar _ehg,
                 bool rejected = false) {
    ehg = _ehg;
    Index nret = 0;
    // obtain the Jacobian matrix
    if (!this->iUserJac) {
      if (this->fdjac(*this->functor, t, u, f, J,
                this->neq - 1, this->neq - 1, this->epsfcn) < 0)
        return -1;
      nret = this->neq;
    } else {
      if (this->functor->df(t, u, J) < 0)
        return -1;
      nret = 1;
    }
    // M = ehg*I - J
    MatrixType M = -J;
    M.diagonal() += VectorType::Constant(this->neq, ehg);
    // LU factorization
    MLU.compute(M);
    return nret;
  }

  Index stageIMPL(Index s,
                  const Eigen::Ref<const VectorType> b,
                  Eigen::Ref<VectorType> x) {
    // (ehg*I - J) x = b
    x.noalias() = MLU.solve(b);
    return 0;
  };

 private:
  MatrixType J;
  Eigen::PartialPivLU <MatrixType> MLU;
  Scalar ehg;

  void resizeWork() {
    this->neq = this->functor->values();
    eigen_assert(this->neq == this->functor->values());
    // resize everything
    J.resize(this->neq, this->neq);
  }
};

template<typename FunctorType, typename Scalar = double>
class ODEJacSAP
    : public ODEJac<ODEJacSAP<FunctorType, Scalar>, FunctorType, Scalar> {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODEJacSAP(FunctorType *_functor, const ODEOptions <Scalar> &_opt)
      : ODEJac<ODEJacSAP<FunctorType, Scalar>, FunctorType, Scalar>
            (_functor, SAP, _opt),
        mk(_opt.maxKryDim),
        MLU(Eigen::PartialPivLU<MatrixType>(this->maxKryDim)) {
    eigen_assert(this->neq == this->functor->values());
    resizeWork();
  }

  Index initIMPL(Scalar t,
                 Eigen::Ref<VectorType> u,
                 Eigen::Ref<VectorType> f,
                 Scalar _ehg,
                 bool rejected = false) {
    resizeWork();
    ehg = _ehg;
    Index nret = 0;
    // obtain the krylov approx. to Jacobian matrix
    if (!rejected) nret = arnoldi(t, u, f);
    M = -H.topLeftCorner(mk, mk);
    M.diagonal() += VectorType::Constant(mk, ehg);
    // LU factorization
    MLU.compute(M);
    return nret;
  }

  Index stageIMPL(Index s,
                  const Eigen::Ref<const VectorType> b,
                  Eigen::Ref<VectorType> x) {
    // (ehg*I - J) x = b
    if (mk != this->maxKryDim) {
      wb1.resize(mk);
      wb2.resize(mk);
    }

    // wb1= Q.leftCols(mk).transpose() * b;
    // wb2 = MLU.solve(wb1);
    // wb1 = Q.leftCols(mk) * wb2;
    // wb1 /= -(ehg * ehg);
    // x = (1.0 / ehg) * b + wb1;

    wb1.noalias() = Q.leftCols(mk).transpose() * b;
    wb2 = MLU.solve(wb1);
    wb1 /= ehg;
    wb2 -= wb1;
    wa1.noalias() = Q.leftCols(mk) * wb2;
    wa2 = b;
    wa2 /= ehg;
    x = wa2 + wa1;

    return 0;
  };

 private:
  Index mk;
  Scalar arnTol;
  VectorType wa1, wb1, wb2, wa2;
  MatrixType H;
  MatrixType Q;
  MatrixType J;
  MatrixType M;
  Eigen::PartialPivLU <MatrixType> MLU;
  Scalar ehg;

  void resizeWork() {
    if (!this->iReSize) return;
    this->iReSize = false;
    this->neq = this->functor->values();
    eigen_assert(this->neq == this->functor->values());

    // resize everything
    wa1.resize(this->neq);
    wa2.resize(this->neq);
    wb1.resize(this->maxKryDim);
    wb2.resize(this->maxKryDim);
    H.resize(this->maxKryDim + 1, this->maxKryDim);
    Q.resize(this->neq, this->maxKryDim + 1);
    H.setZero();
    Q.setZero();
    if (this->iUserJac) {
      J.resize(this->neq, this->neq);
      J.setZero();
    }
  }

  Index arnoldi(Scalar t, Eigen::Ref<VectorType> u, Eigen::Ref<VectorType> f) {
    Index nret = 0;
    Scalar eta = 1.0e4;
    Scalar tau, rho;
    Scalar beta = f.blueNorm();
    Q.col(0) = f;
    Q.col(0) /= std::max(beta, Eigen::NumTraits<Scalar>::epsilon());
    if (this->iUserJac) {
      if (this->functor->df(t, u, J) < 0)
        return -1;
      nret++;
    }
    for (Index i = 0; i < this->maxKryDim; i++) {
      wa1 = Q.col(i);
      if (!this->iUserJac) {
        if (this->fdjacv(*this->functor, t, u, f, wa1, wa2, this->epsfcn) < 0)
          return -1;
        nret++;
      } else {
        wa2 = J * wa1;
      }
      tau = wa2.blueNorm();
      for (Index j = 0; j < i + 1; j++) {
        H(j, i) = wa2.dot(Q.col(j));
        wa2 -= H(j, i) * Q.col(j);
      }
      if (wa2.blueNorm() <= eta * tau) {
        for (Index j = 0; j < i + 1; j++) {
          rho = wa2.dot(Q.col(j));
          wa2 -= rho * Q.col(j);
          H(j, i) += rho;
        }
      }
      H(i + 1, i) = wa2.blueNorm();
      Q.col(i + 1) = wa2 / std::max(H(i + 1, i), Eigen::NumTraits<Scalar>::epsilon());
      if (H(i + 1, i) <= this->arnTol * tau) {
        mk = i + 1;
        return nret;
      }
    }
    mk = this->maxKryDim;
    return nret;
  }
};

template<typename FunctorType, typename Scalar = double>
class ODEJacHAP
    : public ODEJac<ODEJacHAP<FunctorType, Scalar>, FunctorType, Scalar> {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagType;

  ODEJacHAP(FunctorType *_functor, const ODEOptions <Scalar> &_opt)
      : ODEJac<ODEJacHAP<FunctorType, Scalar>, FunctorType, Scalar>
            (_functor, HAP, _opt),
        mk(_opt.maxKryDim),
        MLU(Eigen::PartialPivLU<MatrixType>(this->maxKryDim)) {
    eigen_assert(this->neq == this->functor->values());
    resizeWork();
  }

  Index initIMPL(Scalar t,
                 Eigen::Ref<VectorType> u,
                 Eigen::Ref<VectorType> f,
                 Scalar _ehg,
                 bool rejected = false) {
    resizeWork();
    ehg = _ehg;
    Index nret = 0;
    // obtain the krylov approx. to Jacobian matrix
    if (!rejected) nret = arnoldi(t, u, f);
    MatrixType M = -H.topLeftCorner(mk, mk);
    M.diagonal() += VectorType::Constant(mk, ehg);
    // LU factorization
    MLU.compute(M);
    return nret;
  }

  Index stageIMPL(Index s,
                  const Eigen::Ref<const VectorType> b,
                  Eigen::Ref<VectorType> x) {
    // (ehg*I - J) x = b
    if (mk != this->maxKryDim) {
      wb1.resize(mk);
      wb2.resize(mk);
    }
    wb1.noalias() = Q.leftCols(mk).transpose() * b;
    wb2 = MLU.solve(wb1);
    wb1 /= ehg;
    wb2 -= wb1;
    wa1.noalias() = Q.leftCols(mk) * wb2;
    wa2 = b;
    wa2 /= ehg;
    x = wa2 + wa1;
    return 0;
  };

 private:
  Index mk;
  Scalar arnTol;
  VectorType wa1, wb1, wb2, wa2;
  MatrixType H;
  MatrixType Q;
  MatrixType J;
  Eigen::PartialPivLU <MatrixType> MLU;
  Scalar ehg;

  void resizeWork() {
    if (!this->iReSize) return;
    this->iReSize = false;
    this->neq = this->functor->values();
    eigen_assert(this->neq == this->functor->values());

    // resize everything
    wa1.resize(this->neq);
    wa2.resize(this->neq);
    wb1.resize(this->maxKryDim);
    wb2.resize(this->maxKryDim);
    H.resize(this->maxKryDim + 1, this->maxKryDim);
    Q.resize(this->neq, this->maxKryDim + 1);
    H.setZero();
    Q.setZero();
    if (this->iUserJac) {
      J.resize(this->neq, this->neq);
      J.setZero();
    }
  }

  Index arnoldi(Scalar t, Eigen::Ref<VectorType> u, Eigen::Ref<VectorType> f) {
    Index nret = 0;
    Scalar eta = 1.0e4;
    Scalar tau, rho;
    Scalar beta = f.blueNorm();
    Q.col(0) = f;
    Q.col(0) /= std::max(beta, Eigen::NumTraits<Scalar>::epsilon());
    if (this->iUserJac) {
      if (this->functor->df(t, u, J) < 0)
        return -1;
      nret++;
    }
    for (Index i = 0; i < this->maxKryDim; i++) {
      wa1 = Q.col(i);
      if (!this->iUserJac) {
        if (this->fdjacv(*this->functor, t, u, f, wa1, wa2, this->epsfcn) < 0)
          return -1;
        nret++;
      } else {
        wa2 = J * wa1;
      }
      tau = wa2.blueNorm();
      for (Index j = 0; j < i + 1; j++) {
        H(j, i) = wa2.dot(Q.col(j));
        wa2 -= H(j, i) * Q.col(j);
      }
      if (wa2.blueNorm() <= eta * tau) {
        for (Index j = 0; j < i + 1; j++) {
          rho = wa2.dot(Q.col(j));
          wa2 -= rho * Q.col(j);
          H(j, i) += rho;
        }
      }
      H(i + 1, i) = wa2.blueNorm();
      Q.col(i + 1) = wa2 / std::max(H(i + 1, i),
                                    Eigen::NumTraits<Scalar>::epsilon());
      if (H(i + 1, i) <= this->arnTol * tau) {
        mk = i + 1;
        return nret;
      }
    }
    mk = this->maxKryDim;
    return nret;
  }
};

}

#endif
