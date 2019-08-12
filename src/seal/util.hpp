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

#include <array>
#include <complex>
#include <string>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace he {
static inline void print_seal_context(const seal::SEALContext& context) {
  auto& context_data = *context.key_context_data();

  NGRAPH_CHECK(context_data.parms().scheme() == seal::scheme_type::CKKS,
               "Only CKKS scheme supported");

  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: CKKS" << std::endl;
  std::cout << "|   poly_modulus_degree: "
            << context_data.parms().poly_modulus_degree() << std::endl;
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_mod_count = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_mod_count - 1; i++) {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;
  std::cout << "\\" << std::endl;
}

// Packs elements of input into real values
// (a+bi, c+di) => (a,b,c,d)
template <typename T>
static inline void complex_vec_to_real_vec(
    std::vector<T>& output, const std::vector<std::complex<T>>& input) {
  NGRAPH_CHECK(output.size() == 0);
  output.reserve(input.size() * 2);
  for (const std::complex<T>& value : input) {
    output.emplace_back(value.real());
    output.emplace_back(value.imag());
  }
}

// Packs elements of input into complex values
// (a,b,c,d) => (a+bi, c+di)
// (a,b,c) => (a+bi, c+0i)
template <typename T>
static inline void real_vec_to_complex_vec(std::vector<std::complex<T>>& output,
                                           const std::vector<T>& input) {
  NGRAPH_CHECK(output.size() == 0);
  output.reserve(input.size() / 2);
  std::vector<T> complex_parts(2, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    complex_parts[i % 2] = input[i];

    if (i % 2 == 1 || i == input.size() - 1) {
      output.emplace_back(std::complex<T>(complex_parts[0], complex_parts[1]));
      complex_parts = {T(0), T(0)};
    }
  }
}

static inline bool flag_to_bool(const char* flag, bool default_value = false) {
  if (flag == nullptr) {
    return default_value;
  }
  static std::unordered_set<std::string> on_map{"1", "y", "yes"};
  static std::unordered_set<std::string> off_map{"0", "n", "no"};
  std::string flag_str = ngraph::to_lower(std::string(flag));

  if (on_map.find(flag_str) != on_map.end()) {
    return true;
  } else if (off_map.find(flag_str) != off_map.end()) {
    return true;
  } else {
    throw ngraph_error("Unknown flag value " + std::string(flag));
  }
}

static inline double type_to_double(const void* src,
                                    const element::Type& element_type) {
  switch (element_type.get_type_enum()) {
    case element::Type_t::f32:
      return static_cast<double>(*static_cast<const float*>(src));
      break;
    case element::Type_t::f64:
      return static_cast<double>(*static_cast<const double*>(src));
      break;
    case element::Type_t::i64:
      return static_cast<double>(*static_cast<const int64_t*>(src));
      break;
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::i32:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    case element::Type_t::dynamic:
    case element::Type_t::undefined:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::boolean:
      NGRAPH_CHECK(false, "Unsupported element type ", element_type);
      break;
  }
  NGRAPH_CHECK(false, "Unsupported element type ", element_type);
  return 0.0;
}

static inline std::vector<double> type_vec_to_double_vec(
    const void* src, const element::Type& element_type, size_t n) {
  std::vector<double> ret(n);
  char* src_with_offset = static_cast<char*>(const_cast<void*>(src));
  for (size_t i = 0; i < n; ++i) {
    ret[i] = ngraph::he::type_to_double(src_with_offset, element_type);
    ++src_with_offset;
  }
  return ret;
}

static void double_vec_to_type_vec(void* target,
                                   const element::Type& element_type,
                                   const std::vector<double>& input) {
  NGRAPH_CHECK(input.size() > 0, "Input has no values");
  size_t count = input.size();
  size_t type_byte_size = element_type.size();

  switch (element_type.get_type_enum()) {
    case element::Type_t::f32: {
      std::vector<float> float_values{input.begin(), input.end()};
      void* type_values_src =
          static_cast<void*>(const_cast<float*>(float_values.data()));
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::f64: {
      void* type_values_src =
          static_cast<void*>(const_cast<double*>(input.data()));
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::i64: {
      std::vector<int64_t> int64_values(input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        int64_values[i] = std::round(input[i]);
      }
      void* type_values_src =
          static_cast<void*>(const_cast<int64_t*>(int64_values.data()));
      std::memcpy(target, type_values_src, type_byte_size * count);
      break;
    }
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::i32:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    case element::Type_t::dynamic:
    case element::Type_t::undefined:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::boolean:
      NGRAPH_CHECK(false, "Unsupported element type ", element_type);
      break;
  }
}

inline void save(const seal::Ciphertext& cipher, void* destination) {
  static constexpr std::array<size_t, 6> offsets = {
      sizeof(seal::parms_id_type),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) + sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          2 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t) + sizeof(double),
  };

  char* dst_char = static_cast<char*>(destination);

  bool is_ntt_form = cipher.is_ntt_form();
  uint64_t size = cipher.size();
  uint64_t polynomial_modulus_degree = cipher.poly_modulus_degree();
  uint64_t coeff_mod_count = cipher.coeff_mod_count();

  std::memcpy(destination, (void*)&cipher.parms_id(),
              sizeof(seal::parms_id_type));
  std::memcpy(static_cast<void*>(dst_char + offsets[0]), (void*)&is_ntt_form,
              sizeof(seal::SEAL_BYTE));
  std::memcpy(static_cast<void*>(dst_char + offsets[1]), (void*)&size,
              sizeof(uint64_t));
  std::memcpy(static_cast<void*>(dst_char + offsets[2]),
              (void*)&polynomial_modulus_degree, sizeof(uint64_t));
  std::memcpy(static_cast<void*>(dst_char + offsets[3]),
              (void*)&coeff_mod_count, sizeof(uint64_t));
  std::memcpy(static_cast<void*>(dst_char + offsets[4]), (void*)&cipher.scale(),
              sizeof(double));
  std::memcpy(static_cast<void*>(dst_char + offsets[5]), (void*)cipher.data(),
              8 * cipher.uint64_count());
}

inline void load(seal::Ciphertext& cipher, void* src) {
  seal::parms_id_type parms_id{};
  seal::SEAL_BYTE is_ntt_form_byte;
  uint64_t size64 = 0;
  uint64_t poly_modulus_degree = 0;
  uint64_t coeff_mod_count = 0;
  double scale = 0;

  static constexpr std::array<size_t, 6> offsets = {
      sizeof(seal::parms_id_type),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) + sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          2 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t),
      sizeof(seal::parms_id_type) + sizeof(seal::SEAL_BYTE) +
          3 * sizeof(uint64_t) + sizeof(double),
  };

  char* char_src = static_cast<char*>(src);

  std::memcpy(&parms_id, src, sizeof(seal::parms_id_type));
  std::memcpy(&is_ntt_form_byte, static_cast<void*>(char_src + offsets[0]),
              sizeof(seal::SEAL_BYTE));
  std::memcpy(&size64, static_cast<void*>(char_src + offsets[1]),
              sizeof(uint64_t));
  std::memcpy(&poly_modulus_degree, static_cast<void*>(char_src + offsets[2]),
              sizeof(uint64_t));
  std::memcpy(&coeff_mod_count, static_cast<void*>(char_src + offsets[3]),
              sizeof(uint64_t));
  std::memcpy(&scale, static_cast<void*>(char_src + offsets[4]),
              sizeof(double));

  bool ntt_form = (is_ntt_form_byte == seal::SEAL_BYTE(0)) ? false : true;

  NGRAPH_INFO << "Loaded nttform " << ntt_form;
  NGRAPH_INFO << "loaded size64 " << size64;
  NGRAPH_INFO << "Loaded poly_modulus_degree " << poly_modulus_degree;
  NGRAPH_INFO << "Loaded coeff_mod_count " << coeff_mod_count;
  NGRAPH_INFO << "Loaded scale " << scale;

  size_t data_count = poly_modulus_degree * coeff_mod_count;

  seal::IntArray<seal::Ciphertext::ct_coeff_type> new_data(data_count,
                                                           cipher.pool());

  cipher.is_ntt_form() =
      (is_ntt_form_byte == seal::SEAL_BYTE(0)) ? false : true;
  cipher.scale() = scale;
  cipher.parms_id() = parms_id;

  // TODO: load/ verify context?
  seal::EncryptionParameters parms(seal::scheme_type::CKKS);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  // parms.set_coeff_modulus();
  auto context = seal::SEALContext::Create(parms);

  cipher.reserve(size64);
  cipher.resize(size64);

  NGRAPH_INFO << "cipher size " << cipher.size();
  NGRAPH_INFO << "Copying to new data";
  NGRAPH_INFO << "data_count " << data_count;
  NGRAPH_INFO << "sizeof(seal::Ciphertext::ct_coeff_type) "
              << sizeof(seal::Ciphertext::ct_coeff_type);
  std::memcpy(static_cast<void*>(new_data.begin()),
              static_cast<void*>(char_src + offsets[5]),
              sizeof(seal::Ciphertext::ct_coeff_type) * data_count);
  NGRAPH_INFO << "copying to new_data.data() is fine";

  std::memcpy(static_cast<void*>(cipher.data()),
              static_cast<void*>(char_src + offsets[5]),
              sizeof(seal::Ciphertext::ct_coeff_type) * data_count);

  // std::memcpy(destination, (void*)cipher.data()), 8 *
  // cipher.uint64_count());
}

}  // namespace he
}  // namespace ngraph