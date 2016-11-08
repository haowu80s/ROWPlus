//
// Created by Hao Wu on 11/8/16.
//

#ifndef ROWPLUS_STEPPER_H
#define ROWPLUS_STEPPER_H

#include "ROWPlus/Core/ODEUtility.h"

namespace ROWPlus {

template<typename Scalar = double>
class Stepper {
 private:
  ODEOptions<Scalar> m_options;
};

template<typename Scalar = double>
class rosenbrock_krylov4 {
};

}


#endif //ROWPLUS_STEPPER_H
