//
// Created by Hao Wu on 11/10/16.
//

#ifndef ROWPLUS_RXNFUNCTOR_H
#define ROWPLUS_RXNFUNCTOR_H

#include "ROWPlus/ROWPlus.h"
#include <Eigen/Core>

#include "cantera/IdealGasMix.h"
#include "cantera/transport.h"
#include "cantera/transport/MixEzTransport.h"

class RxnFunctor {
 public:
  typedef std::vector<double> state_type;
  RxnFunctor(const Cantera::IdealGasMix &_gas);

  void operator()(const state_type &x, state_type &dxdt, const double t);

  int f(double t, const Eigen::Ref<const Eigen::VectorXd> x,
        Eigen::Ref<Eigen::VectorXd> v);

  int fdt(double t, const Eigen::Ref<const Eigen::VectorXd> x,
          Eigen::Ref<Eigen::VectorXd> v) {
    throw std::runtime_error("RxnFunctor::fdt() is not implemented.");
    return -1;
  }

  int df(double t,
         const Eigen::Ref<const Eigen::VectorXd> x,
         Eigen::Ref<Eigen::MatrixXd> m) {
    throw std::runtime_error("RxnFunctor::fdt() is not implemented.");
    return -1;
  }

  int dfd(double t,
          const Eigen::Ref<const Eigen::VectorXd> x,
          Eigen::Ref<Eigen::VectorXd> v);

  void checkBound(bool _check) { m_check_bnd = _check; }

  Eigen::DenseIndex values() const { return m_nsp + 1; }
  Eigen::DenseIndex inputs() const { return m_nsp + 1; }
 private:
  Cantera::IdealGasMix m_gas;
  const size_t m_nsp;
  Eigen::VectorXd m_wa1, m_wa2;
  bool m_check_bnd;

  int evalEqsConstVol(const Eigen::Ref<const Eigen::VectorXd> x,
                      Eigen::Ref<Eigen::VectorXd> v);

  int evalEqsConstVolDiagJac(const Eigen::Ref<const Eigen::VectorXd> x,
                             Eigen::Ref<Eigen::VectorXd> v);
};

#endif //ROWPLUS_RXNFUNCTOR_H
