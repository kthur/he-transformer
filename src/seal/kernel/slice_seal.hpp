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

#include <vector>

#include "he_type.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph::he {
inline void slice_seal(const std::vector<HEType>& arg, std::vector<HEType>& out,
                       const Shape& arg_shape, const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds, const Strides& strides,
                       const Shape& out_shape) {
  CoordinateTransform input_transform(arg_shape, lower_bounds, upper_bounds,
                                      strides);
  CoordinateTransform output_transform(out_shape);

  CoordinateTransform::Iterator output_it = output_transform.begin();

  NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                   shape_size(output_transform.get_target_shape()),
               "Slice transform shape sizes don't match");

  for (const Coordinate& in_coord : input_transform) {
    const Coordinate& out_coord = *output_it;

    out[output_transform.index(out_coord)] =
        arg[input_transform.index(in_coord)];

    ++output_it;
  }
}

}  // namespace ngraph::he
