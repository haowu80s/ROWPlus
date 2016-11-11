//
// Created by Hao Wu on 11/10/16.
//

#include "RxnFunctor.h"

using namespace Eigen;
using namespace Cantera;
using namespace std;

RxnFunctor::RxnFunctor(const IdealGasMix &_gas) :
    m_gas(_gas), m_nsp(m_gas.nSpecies()), m_wa1(m_nsp), m_wa2(m_nsp),
    m_check_bnd(false) {}

void RxnFunctor::operator()(const RxnFunctor::state_type &x,
                            RxnFunctor::state_type &dxdt,
                            const double t) {
  Map<const VectorXd> xvec(x.data(), x.size());
  Map<VectorXd> fvec(dxdt.data(), dxdt.size());
  evalEqsConstVol(xvec, fvec);
}

int RxnFunctor::f(double t,
                  const Ref<const VectorXd> x,
                  Ref<VectorXd> fvec) {
  static const double tol = -std::sqrt(Eigen::NumTraits<double>::epsilon());
  if (m_check_bnd && x.minCoeff() < tol) return -1;
  return evalEqsConstVol(x, fvec);
}

int RxnFunctor::evalEqsConstVol(const Ref<const Eigen::VectorXd> x,
                                Ref<Eigen::VectorXd> fvec) {
  // The components of y are [0] the temperature,
  // [1...K+1) are the mass fractions of each species
  try {
    // update state
    m_gas.setState_TRY(x[0], m_gas.density(), x.data() + 1);
    // fetch molecular weights [kg/kmol]
    const Cantera::vector_fp &mw = m_gas.molecularWeights();
    // compute partial molar intEnergies [J/kmol]
    m_gas.getPartialMolarIntEnergies(m_wa1.data());
    // compute net production rates [kmol/m^3/s]
    m_gas.getNetProductionRates(m_wa2.data()); // "omega dot"
    // convert to [(kg/kg)/s]
    fvec.tail(m_nsp) =
        m_wa2.cwiseProduct(Map<const VectorXd>(mw.data(), mw.size())) / m_gas.density();
    // compute dT/dt [K/s]
    fvec(0) = -m_wa2.dot(m_wa1) / m_gas.density() / m_gas.cv_mass();
    return 0;
  }
  catch (...) { return -1; }
}