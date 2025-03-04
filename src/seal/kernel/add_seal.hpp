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

#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::he {
/// \brief Adds two ciphertexts
/// \param[in,out] arg0 Ciphertext argument to add. May be rescaled
/// \param[in,out] arg1 Ciphertext argument to add. May be rescaled
/// \param[out] out Stores the encrypted sum
/// \param[in] he_seal_backend Backend used to perform addition
/// \param[in] pool Memory pool used for new memory allocation
void scalar_add_seal(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    std::shared_ptr<SealCiphertextWrapper>& out, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool = seal::MemoryManager::GetPool());

/// \brief Adds a ciphertext with a plaintext
/// \param[in,out] arg0 Ciphertext argument to add. May be rescaled
/// \param[in] arg1 Plaintext argument to add
/// \param[in] complex_packing Whether or not the ciphertext should be added
/// using complex packing
/// \param[out] out Stores the encrypted sum
/// \param[in] he_seal_backend Backend used to perform addition
void scalar_add_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const bool complex_packing,
                     HESealBackend& he_seal_backend);

/// \brief Adds a plaintext with a ciphertext
/// \param[in] arg0 Plaintext argument to add
/// \param[in,out] arg1 Ciphertext argument to add. May be rescaled
/// \param[out] out Stores the encrypted sum
/// \param[in] complex_packing Whether or not the ciphertext should be added
/// using complex packing
/// \param[in] he_seal_backend Backend used to perform addition
inline void scalar_add_seal(const HEPlaintext& arg0,
                            SealCiphertextWrapper& arg1,
                            std::shared_ptr<SealCiphertextWrapper>& out,
                            const bool complex_packing,
                            HESealBackend& he_seal_backend) {
  scalar_add_seal(arg1, arg0, out, complex_packing, he_seal_backend);
}

/// \brief Adds two plaintexts
/// \param[in] arg0 Plaintext argument to add
/// \param[in] arg1 Plaintext argument to add
/// \param[out] out Stores the plaintext sum
void scalar_add_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                     HEPlaintext& out);

/// \brief Adds two ciphertext/plaintext elements
/// \param[in] arg0 Cipher or plaintext data to add
/// \param[in] arg1 Cipher or plaintext data to add
/// \param[in] out Stores the ciphertext or plaintext sum
/// \param[in] he_seal_backend Backend used to perform addition
inline void scalar_add_seal(HEType& arg0, HEType& arg1, HEType& out,
                            HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
               "Complex packing types don't match");
  out.complex_packing() = arg0.complex_packing();

  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg0.get_ciphertext(), *arg1.get_ciphertext(),
                    out.get_ciphertext(), he_seal_backend);
  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg0.get_ciphertext(), arg1.get_plaintext(),
                    out.get_ciphertext(), arg0.complex_packing(),
                    he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg1.get_ciphertext(), arg0.get_plaintext(),
                    out.get_ciphertext(), arg0.complex_packing(),
                    he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    if (!out.is_plaintext()) {
      out.set_plaintext(HEPlaintext());
    }
    scalar_add_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                    out.get_plaintext());
  } else {
    NGRAPH_CHECK(false, "Unknown argument types");
  }
}

/// \brief Adds two vectors of ciphertext/plaintext elements element-wise
/// \param[in] arg0 Cipher or plaintext data to add
/// \param[in] arg1 Cipher or plaintext data to add
/// \param[in] out Stores the ciphertext or plaintext sum
/// \param[in] count Number of elements to add
/// \param[in] element_type datatype of the data to add
/// \param[in] he_seal_backend Backend used to perform addition
inline void add_seal(std::vector<HEType>& arg0, std::vector<HEType>& arg1,
                     std::vector<HEType>& out, size_t count,
                     const element::Type& element_type,
                     HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
               "Unsupported type ", element_type);
  NGRAPH_CHECK(count <= arg0.size(), "Count ", count,
               " is too large for arg0, with size ", arg0.size());
  NGRAPH_CHECK(count <= arg1.size(), "Count ", count,
               " is too large for arg1, with size ", arg1.size());

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    scalar_add_seal(arg0[i], arg1[i], out[i], he_seal_backend);
  }
}

}  // namespace ngraph::he
