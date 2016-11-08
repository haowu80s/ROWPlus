//
// Created by Hao Wu on 11/7/16.
//

#ifndef ROWPLUS_LORENZ96_H
#define ROWPLUS_LORENZ96_H

#include "ROWPlus/ROWPlus.h"
#include <Eigen/Core>

class lorenz96 : public ROWPlus::BaseFunctor<double, 40, 40> {
 public:
  typedef std::vector< double > state_type;

  lorenz96(double F = 8.0);

  void operator() ( const state_type &x , state_type &dxdt , const double t );

  Eigen::DenseIndex f(double t, const Eigen::Ref<const Eigen::VectorXd> &x,
                      Eigen::VectorXd &v);

  Eigen::DenseIndex fdt(double t, const Eigen::Ref<const Eigen::VectorXd> &x,
                        Eigen::VectorXd &v);

  Eigen::DenseIndex df(double t,
                       const Eigen::Ref<const Eigen::VectorXd> &x,
                       Eigen::MatrixXd &m);

  bool checkPositivity(const Eigen::Ref<const Eigen::VectorXd> &x) { return true; };

 private:
  const double m_F;
  InputType m_wa1, m_wa2;
};

#endif //ROWPLUS_LORENZ96_H
