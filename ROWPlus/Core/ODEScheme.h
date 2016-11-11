#ifndef ODE_SCHEME_H
#define ODE_SCHEME_H

#include <stdexcept>
#include <vector>
#include <iostream>

#include <Eigen/Core>

#include "ROWPlus/Core/ODEUtility.h"

namespace ROWPlus {

using std::unique_ptr;
using std::vector;

template<typename Scalar = double>
class ODEScheme {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  ODEScheme(const MatrixType &aij_, const MatrixType &gij_,
            const MatrixType &cij_, const VectorType &bi_,
            const VectorType &ci_, const VectorType &di_, const VectorType &ei_,
            Scalar gamma_, const vector<int> _cf, Index nStage_, Index nOrder_)
      : aij(aij_), gij(gij_), cij(cij_), bi(bi_), ci(ci_), di(di_), ei(ei_),
        gamma(gamma_), cf(_cf), nStage(nStage_), nOrder(nOrder_) {}

  // Coeff. of the scheme
  const MatrixType aij;
  const MatrixType gij;
  const MatrixType cij;
  const VectorType bi;
  const VectorType ci;
  const VectorType di;
  const VectorType ei;
  const Scalar gamma;
  const vector<int> cf;
  const Index nStage;
  const Index nOrder;

  // proot
  Scalar proot(Scalar x) {
    switch (nOrder) {
      case (4):return sqrt(sqrt(x));
        break;
      case (3):return cbrt(x);
        break;
      case (2):return sqrt(x);
        break;
      default:return pow(x, 1.0 / (Scalar) nOrder);
        break;
    }
  }
  Scalar pproot(Scalar x) {
    switch (nOrder) {
      case (4):return cbrt(sqrt(x));
        break;
      case (3):return sqrt(sqrt(x));
        break;
      case (2):return cbrt(x);
        break;
      default:return pow(x, 1.0 / (Scalar) (nOrder + 1));
        break;
    }
  }
};

template<typename Scalar = double>
class ODESchemeFactory {
 public:
  typedef Eigen::DenseIndex Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  // Factory Method
  static unique_ptr<ODEScheme<Scalar> > make_ODEScheme(ODESchemeType choice);
};

template<typename Scalar>
unique_ptr<ODEScheme<Scalar> >
ODESchemeFactory<Scalar>::make_ODEScheme(ODESchemeType choice) {
  MatrixType aij;
  MatrixType gij;
  MatrixType cij;
  VectorType bi;
  VectorType ci;
  VectorType di;
  VectorType ei;
  Scalar gamma;
  vector<int> cf;
  Index nStage;
  Index nOrder;

  switch (choice) {
    case SHAMP:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 1.00000000000000, 0.00000000000000,
          0.48000000000000, 0.12000000000000;
      gij.resize(3, 3);
      gij << -2.00000000000000, 0.00000000000000, 0.00000000000000,
          1.32000000000000, 0.60000000000000, 0.00000000000000,
          -0.05600000000000, -0.22800000000000, -0.10000000000000;
      bi.resize(4);
      bi << 0.29629629629630, 0.12500000000000, 0.0, 0.57870370370370;
      ei.resize(4);
      ei << 0, -0.04166666666667, -0.11574074074074, 1.15740740740741;
      gamma = 0.50000000000000;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case GRK4T:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 0.46200000000000, 0.00000000000000, -0.08156681683272,
          0.96177515016606;
      gij.resize(3, 3);
      gij << -0.27062966775244, 0.00000000000000, 0.00000000000000,
          0.31125448329409, 0.00852445628482, 0.00000000000000, 0.28281683204353,
          -0.45795948328073, -0.11120833333333;
      bi.resize(4);
      bi << 0.21748737165273, 0.48622903799012, 0.0, 0.29628359035715;
      ei.resize(4);
      ei << -0.71708850449933, 1.77617912176104, -0.05909061726171,
          0.00000000000000;
      gamma = 0.23100000000000;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case VELDS:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 1.00000000000000, 0.00000000000000, 0.37500000000000,
          0.12500000000000;
      gij.resize(3, 3);
      gij << -2.00000000000000, 0.00000000000000, 0.00000000000000,
          -1.00000000000000, -0.25000000000000, 0.00000000000000,
          -0.37500000000000, -0.37500000000000, 0.50000000000000;
      bi.resize(4);
      bi << 0.16666666666667, 0.16666666666667, 0.0, 0.66666666666667;
      ei.resize(4);
      ei << 1.16666666666667, 0.50000000000000, -0.66666666666667,
          0.00000000000000;
      gamma = 0.50000000000000;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case VELDD:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 0.45141622964514, 0.00000000000000, -0.15773202438639,
          1.03332491898823;
      gij.resize(3, 3);
      gij << -0.27170214984937, 0.00000000000000, 0.00000000000000,
          0.20011014796684, 0.09194078770500, 0.00000000000000, 0.35990464608231,
          -0.52236799086101, -0.10130100942441;
      bi.resize(4);
      bi << 0.20961757675658, 0.48433148684810, 0.0, 0.30605093639532;
      ei.resize(4);
      ei << -0.74638173030838, 1.78642253324799, -0.04004080293962,
          0.00000000000000;
      gamma = 0.22570811482257;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case LSTAB:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 1.14564000000000, 0.00000000000000, 0.52092209544722,
          0.13429476836837;
      gij.resize(3, 3);
      gij << -2.34201389131923, 0.00000000000000, 0.00000000000000,
          -0.02735980356646, 0.21380314735851, 0.00000000000000,
          -0.25909062216449, -0.19059462272997, -0.22803686381559;
      bi.resize(4);
      bi << 0.32453574762832, 0.04908429214667, 0.0, 0.62637996022502;
      ei.resize(4);
      ei << 0.61994881642181, 0.19268272217757, 0.18736846140061,
          0.00000000000000;
      gamma = 0.572816062482135;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case GRK4A:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 0.43800000000000, 0.00000000000000, 0.79692045793846,
          0.07307954206154;
      gij.resize(3, 3);
      gij << -0.76767239548409, 0.00000000000000, 0.00000000000000,
          -0.85167532374233, 0.52296728918805, 0.00000000000000, 0.28846310954547,
          0.08802142733812, -0.33738984062673;
      bi.resize(4);
      bi << 0.19929327570063, 0.48264523567374, 0.06806148862563,
          0.25000000000000;
      ei.resize(4);
      ei << 0.34632583375795, 0.28569317571228, 0.36798099052978,
          0.00000000000000;
      gamma = 0.39500000000000;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case ROK4A:nStage = 4;
      nOrder = 4;

      aij.resize(3, 3);
      aij << 1.00000000000000, 0.00000000000000, 0.00000000000000,
          0.10845300169319, 0.39154699830681, 0.00000000000000, 0.43453047756004,
          0.14484349252001, -0.0793739700800;
      gij.resize(3, 3);
      gij << -1.91153192976055, 0.00000000000000, 0.00000000000000,
          0.32881824061154, 0.00000000000000, 0.00000000000000, 0.03303644239796,
          -0.24375152376108, -0.17062602991994;
      bi.resize(4);
      bi << 0.16666666666667, 0.16666666666667, 0.00000000000000,
          0.66666666666666;
      ei.resize(4);
      ei << 0.50269322573684, 0.27867551969006, 0.21863125457310,
          0.00000000000000;
      gamma = 0.57281606248214;
      cf = {true, true, true, true};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(3);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1), aij(2, 0) + aij(2, 1) + aij(2, 2);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case ROK4E:nStage = 4;
      nOrder = 4;

      aij.resize(2, 2);
      aij << 0.432364435748567, 0.000000000000000, -0.514211316876170,
          1.382271144617360;
      gij.resize(3, 3);
      gij << -0.602765307997356, 0.000000000000000, 0.000000000000000,
          -1.389195789724843, 1.072950969011413, 0.000000000000000,
          0.992356412977094, -1.390032613873701, -0.440875890223325;
      bi.resize(4);
      bi << 0.194335256262729, 0.483167813989227, 0.000000000000000,
          0.322496929748044;
      ei.resize(4);
      ei << -0.217819895945721, 1.031308474784673, 0.186511421161047,
          0.000000000000000;
      gamma = 0.572816062482135;
      cf = {true, true, true, false};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(2);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
    case RKDP:nStage = 7;
      nOrder = 5;

      aij.resize(6, 6);
      aij <<
          1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0,
          44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0,
          19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
            0.0, 0.0,
          9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
            -5103.0 / 18656.0, 0.0,
          35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
            11.0 / 84.0;

      gij.setZero(6, 6);

      bi.resize(7);
      bi << 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0;
      ei.resize(7);
      ei << 5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
          -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0;
      gamma = 1.0;
      cf = {true, true, true, true, true, true, true};
      // common for 7-stage method
      cij.resize(6, 6);
      cij.noalias() = gij / gamma;
      ci.resize(6);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1), aij(2, 0) + aij(2, 1) + aij(2, 2),
          aij(3, 0) + aij(3, 1) + aij(3, 2) + aij(3, 3),
          aij(4, 0) + aij(4, 1) + aij(4, 2) + aij(4, 3) + aij(4, 4),
          aij(5, 0) + aij(5, 1) + aij(5, 2) + aij(5, 3) + aij(5, 4) + aij(5, 5);
      di.resize(7);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2),
          gamma + gij(3, 0) + gij(3, 1) + gij(3, 2) + gij(3, 3),
          gamma + gij(4, 0) + gij(4, 1) + gij(4, 2) + gij(4, 3) + gij(4, 4),
          gamma + gij(5, 0) + gij(5, 1) + gij(5, 2) + gij(5, 3) + gij(5, 4) +
              gij(5, 5);
      ei = ei - bi;
      break;
    case RK23:nStage = 4;
      nOrder = 3;

      aij.resize(3, 3);
      aij <<
          0.5, 0.0, 0.0,
          0.0, 3.0 / 4.0, 0.0,
          2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0;
      gij.Zero(1, 1);
      bi.resize(4);
      bi << 2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0;
      ei.resize(4);
      ei << 7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0;
      gamma = 1.0;
      cf = {false, true, true, true};
      // common for 3-stage method
      cij.Zero(3, 3);
      ci.resize(3);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1), aij(2, 0) + aij(2, 1) + aij(2, 2);
      di = VectorType::Zero(4);
      ei = ei - bi;
      break;
    case ZED34:nStage = 4;
      nOrder = 4;
      gamma = 0.435866521508459;
      aij.resize(3, 3);
      aij << 0.419056148601532, 0.000000000000000, 0.000000000000000,
          0.072934844260333, 0.427065155739667, 0.000000000000000,
          0.397719177305627, -2.456759657677684, 3.059040480372057;
      gij.resize(3, 3);
      gij << -0.636665240575498, 0.000000000000000, 0.000000000000000,
          -0.345741991590889, -0.308057790671799, 0.000000000000000,
          0.653394565920136, 8.405645914451922, -9.059040480372058;
      bi.resize(4);
      bi << 0.397719177305627, -2.456759657677684, 3.059040480372057,
          0.000000000000000;
      ei.resize(4);
      ei << 0.166666666666667, 0.000000000000000, 0.666666666666667,
          0.166666666666667;

      cf = {false, true, true, true};
      // common for 4-stage 4-th order method
      cij.resize(3, 3);
      cij.noalias() = gij / gamma;
      ci.resize(3);
      ci << aij(0, 0), aij(1, 0) + aij(1, 1), aij(2, 0) + aij(2, 1) + aij(2, 2);
      di.resize(4);
      di << gamma, gamma + gij(0, 0), gamma + gij(1, 0) + gij(1, 1),
          gamma + gij(2, 0) + gij(2, 1) + gij(2, 2);
      ei = ei - bi;
      break;
  }
  return unique_ptr<ODEScheme<Scalar> >(new ODEScheme<Scalar>(
      aij, gij, cij, bi, ci, di, ei, gamma, cf, nStage, nOrder));
}
}

#endif
