//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/pad.hpp"
#include "seal/he_seal_backend.hpp"

namespace ngraph::he {
void pad_seal(std::vector<HEType>& arg0,
              std::vector<HEType>& arg1,  // scalar
              std::vector<HEType>& out, const Shape& arg0_shape,
              const Shape& out_shape, const CoordinateDiff& padding_below,
              const CoordinateDiff& padding_above, op::PadMode pad_mode);

}  // namespace ngraph::he
