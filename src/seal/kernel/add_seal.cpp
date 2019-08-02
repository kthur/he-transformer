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

#include "seal/kernel/add_seal.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

void ngraph::he::scalar_add_seal(
    ngraph::he::SealCiphertextWrapper& arg0,
    ngraph::he::SealCiphertextWrapper& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  if (arg0.known_value() && arg1.known_value()) {
    out->known_value() = true;
    out->value() = arg0.value() + arg1.value();
  } else if (arg0.known_value()) {
    HEPlaintext p(arg0.value());
    scalar_add_seal(p, arg1, out, element_type, he_seal_backend, pool);
    out->known_value() = false;
  } else if (arg1.known_value()) {
    HEPlaintext p(arg1.value());
    scalar_add_seal(p, arg0, out, element_type, he_seal_backend, pool);
    out->known_value() = false;
  } else {
    NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
                 "arg0.complex_packing() (", arg0.complex_packing(),
                 ") does not match arg1.complex_packing() (",
                 arg1.complex_packing(), ")");
    NGRAPH_CHECK(arg0.complex_packing() == he_seal_backend.complex_packing(),
                 "Add arg0 is not he_seal_backend.complex_packing()");
    NGRAPH_CHECK(arg1.complex_packing() == he_seal_backend.complex_packing(),
                 "Add arg1 is not he_seal_backend.complex_packing()");

    match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);
    he_seal_backend.get_evaluator()->add(arg0.ciphertext(), arg1.ciphertext(),
                                         out->ciphertext());

    out->known_value() = false;
  }
  out->complex_packing() = he_seal_backend.complex_packing();
}

void ngraph::he::scalar_add_seal(
    ngraph::he::SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
    std::shared_ptr<ngraph::he::SealCiphertextWrapper>& out,
    const element::Type& element_type, HESealBackend& he_seal_backend,
    const seal::MemoryPoolHandle& pool) {
  NGRAPH_CHECK(element_type == element::f32);

  if (arg0.known_value()) {
    out->known_value() = true;
    out->value() = arg0.value() + arg1;
    out->complex_packing() = arg0.complex_packing();
    return;
  }

  // TODO: handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = (arg1 == 0.0f);

  if (add_zero) {
    SealCiphertextWrapper tmp(arg0);
    NGRAPH_CHECK(tmp.complex_packing() == arg0.complex_packing());
    out = std::make_shared<ngraph::he::SealCiphertextWrapper>(tmp);
    out->complex_packing() = tmp.complex_packing();

  } else {
    bool complex_packing = arg0.complex_packing();
    // TODO: optimize for adding single complex number
    if (!complex_packing) {
      float value = arg1;
      double double_val = double(value);
      add_plain(arg0.ciphertext(), double_val, out->ciphertext(),
                he_seal_backend);
    } else {
      auto p = SealPlaintextWrapper(complex_packing);
      ngraph::he::encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
                         arg0.ciphertext().parms_id(),
                         arg0.ciphertext().scale(), complex_packing);
      size_t chain_ind0 = get_chain_index(arg0, he_seal_backend);
      size_t chain_ind1 = get_chain_index(p.plaintext(), he_seal_backend);
      NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
                   chain_ind1, " don't match");

      he_seal_backend.get_evaluator()->add_plain(
          arg0.ciphertext(), p.plaintext(), out->ciphertext());
    }
    out->complex_packing() = arg0.complex_packing();
  }
  out->known_value() = false;
}

void ngraph::he::scalar_add_seal(const HEPlaintext& arg0,
                                 const HEPlaintext& arg1, HEPlaintext& out,
                                 const element::Type& element_type,
                                 HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(element_type == element::f32);
  out = arg0 + arg1;
}
