cmake_minimum_required(VERSION 3.20)
project(MyESL2 VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/fasta_parser.cpp
    src/encoder.cpp
    src/regression.cpp
    src/sg_lasso_fp32.cpp
    src/sg_lasso_fp64.cpp
    src/sg_lasso_leastr_fp32.cpp
    src/sg_lasso_leastr_fp64.cpp
    src/overlapping_sg_lasso_leastr_fp32.cpp
    src/overlapping_sg_lasso_leastr_fp64.cpp
    src/overlapping_sg_lasso_logisticr_fp32.cpp
    src/overlapping_sg_lasso_logisticr_fp64.cpp
    src/gl_logisticr_fp32.cpp
    src/gl_logisticr_fp64.cpp
    src/numeric_parser.cpp
    src/visualizer.cpp
    src/newick.cpp
    src/pipeline_preprocess.cpp
    src/pipeline_encode.cpp
    src/pipeline_train.cpp
    src/pipeline_evaluate.cpp
    src/pipeline_adaptive.cpp
    src/pipeline_psc.cpp
)

# Create executable
add_executable(myesl2 ${SOURCES})

# Link pre-compiled OpenBLAS and LAPACK from deps/
if(WIN32)
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/windows-x64)
    target_link_libraries(myesl2 PRIVATE
        ${DEPS_DIR}/openblas.lib
        ${DEPS_DIR}/lapack.lib
    )
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/openblas.dll
            ${DEPS_DIR}/lapack.dll
            $<TARGET_FILE_DIR:myesl2>
    )
elseif(APPLE)
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/macos-arm64)
    target_link_libraries(myesl2 PRIVATE ${DEPS_DIR}/libopenblas.dylib)
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/libopenblas.dylib
            $<TARGET_FILE_DIR:myesl2>
    )
else()
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/linux-x64)
    target_link_libraries(myesl2 PRIVATE ${DEPS_DIR}/libopenblas.so)
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/libopenblas.so
            $<TARGET_FILE_DIR:myesl2>
    )
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(myesl2 PRIVATE /W4)
else()
    target_compile_options(myesl2 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ----------------------------------------------------------------------------
# Standalone solver comparison test binaries (optional)
#
# These binaries exercise individual solvers in isolation from the FASTA
# encoding pipeline so outputs can be diffed directly against MyESL's
# reference binaries. Gated so CI/release builds can skip them.
# ----------------------------------------------------------------------------

option(BUILD_SOLVER_TESTS "Build standalone solver comparison test binaries" ON)
if(BUILD_SOLVER_TESTS)
    add_executable(gl_logisticr_compare
        test/gl_logisticr_compare.cpp
        src/gl_logisticr_fp64.cpp
    )
    target_include_directories(gl_logisticr_compare PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(WIN32)
        target_link_libraries(gl_logisticr_compare PRIVATE
            ${DEPS_DIR}/openblas.lib ${DEPS_DIR}/lapack.lib)
        add_custom_command(TARGET gl_logisticr_compare POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DEPS_DIR}/openblas.dll ${DEPS_DIR}/lapack.dll
                $<TARGET_FILE_DIR:gl_logisticr_compare>
        )
    elseif(APPLE)
        target_link_libraries(gl_logisticr_compare PRIVATE ${DEPS_DIR}/libopenblas.dylib)
    else()
        target_link_libraries(gl_logisticr_compare PRIVATE ${DEPS_DIR}/libopenblas.so)
    endif()
    if(MSVC)
        target_compile_options(gl_logisticr_compare PRIVATE /W4)
    else()
        target_compile_options(gl_logisticr_compare PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
