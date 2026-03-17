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
    src/numeric_parser.cpp
    src/visualizer.cpp
    src/newick.cpp
    src/pipeline_preprocess.cpp
    src/pipeline_encode.cpp
    src/pipeline_train.cpp
    src/pipeline_evaluate.cpp
)

# Create executable
add_executable(myesl2 ${SOURCES})

# Link OpenBLAS and LAPACK (libraries bundled in include/)
target_link_libraries(myesl2 PRIVATE
    ${PROJECT_SOURCE_DIR}/include/libopenblas.lib
    ${PROJECT_SOURCE_DIR}/include/liblapack.lib
)

# Copy DLLs next to the executable so they are found at runtime
add_custom_command(TARGET myesl2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/include/libopenblas.dll
        ${PROJECT_SOURCE_DIR}/include/liblapack.dll
        $<TARGET_FILE_DIR:myesl2>
)

# Compiler warnings
if(MSVC)
    target_compile_options(myesl2 PRIVATE /W4)
else()
    target_compile_options(myesl2 PRIVATE -Wall -Wextra -Wpedantic)
endif()
