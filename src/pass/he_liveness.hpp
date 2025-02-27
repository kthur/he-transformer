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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph::he::pass {

/// \brief An aggressive version of Liveness which will delete the parameter
/// node and any constant nodes
class HELiveness : public ngraph::pass::FunctionPass {
 public:
  /// \brief Performs HELiveness pass on given function
  /// \param[in,out] function Function to perform pass on
  /// \returns false, indicating the function has not been modified
  bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};
}  // namespace ngraph::he::pass
