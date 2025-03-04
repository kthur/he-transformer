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

#include "he_plaintext.hpp"
#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::he {

void scalar_minimum_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                         HEPlaintext& out);

void scalar_minimum_seal(HEType& arg0, HEType& arg1, HEType& out,
                         HESealBackend& he_seal_backend);

void minimum_seal(const std::vector<HEType>& arg0,
                  const std::vector<HEType>& arg1, std::vector<HEType>& out,
                  size_t count, HESealBackend& he_seal_backend);

}  // namespace ngraph::he
