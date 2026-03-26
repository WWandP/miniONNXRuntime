cmake_minimum_required(VERSION 3.20)

project(minionnxruntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MINIORT_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(MINIORT_ONNX_DIR "${MINIORT_THIRD_PARTY_DIR}/onnx")
set(MINIORT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

foreach(path IN ITEMS
    "${MINIORT_ONNX_DIR}/LICENSE"
    "${MINIORT_ONNX_DIR}/onnx/onnx_pb.h"
    "${MINIORT_ONNX_DIR}/onnx/onnx-ml.proto")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Required dependency path does not exist: ${path}")
  endif()
endforeach()

find_package(Protobuf CONFIG REQUIRED)

add_library(miniort_onnx_proto STATIC
  "${MINIORT_GENERATED_DIR}/onnx/onnx-ml.pb.cc"
)

add_custom_command(
  OUTPUT
    "${MINIORT_GENERATED_DIR}/onnx/onnx-ml.pb.h"
    "${MINIORT_GENERATED_DIR}/onnx/onnx-ml.pb.cc"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MINIORT_GENERATED_DIR}/onnx"
  COMMAND "$<TARGET_FILE:protobuf::protoc>"
          --proto_path=third_party/onnx
          --cpp_out=${MINIORT_GENERATED_DIR}
          third_party/onnx/onnx/onnx-ml.proto
  DEPENDS
    "${MINIORT_ONNX_DIR}/onnx/onnx-ml.proto"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM
)

target_include_directories(miniort_onnx_proto
  PUBLIC
    "${MINIORT_ONNX_DIR}"
    "${MINIORT_GENERATED_DIR}"
)

target_compile_definitions(miniort_onnx_proto PUBLIC ONNX_ML=1)

target_link_libraries(miniort_onnx_proto
  PUBLIC
    protobuf::libprotobuf
)

add_library(miniort_model STATIC
  src/model/graph.cc
  src/loader/onnx_loader.cc
)

target_include_directories(miniort_model
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${MINIORT_ONNX_DIR}"
    "${MINIORT_GENERATED_DIR}"
)

target_compile_definitions(miniort_model PUBLIC ONNX_ML=1)

target_link_libraries(miniort_model
  PUBLIC
    miniort_onnx_proto
)

add_executable(miniort_inspect
  tools/inspect_model.cc
)

target_link_libraries(miniort_inspect PRIVATE miniort_model)
