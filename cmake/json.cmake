# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ******************************************************************************

include(ExternalProject)

set(JSON_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/json)
set(JSON_SRC_DIR ${JSON_PREFIX}/src/ext_json)

message("JSON_PREFIX ${JSON_PREFIX}")
message("JSON_SRC_DIR ${JSON_SRC_DIR}")

set(JSON_GIT_REPO_URL https://github.com/nlohmann/json)
set(JSON_GIT_LABEL v3.5.0)

ExternalProject_Add(ext_json
                    PREFIX json
                    GIT_REPOSITORY ${JSON_GIT_REPO_URL}
                    GIT_TAG ${JSON_GIT_LABEL}
                    # Disable install step
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND cp
                                  -r
                                  ${JSON_SRC_DIR}/single_include/nlohmann
                                  ${EXTERNAL_INSTALL_DIR}/include/
                    INSTALL_COMMAND ""
                                    # INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
                                    # UPDATE_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE)

# ------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_json SOURCE_DIR)
add_library(libjson INTERFACE)
target_include_directories(libjson SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libjson ext_json)
