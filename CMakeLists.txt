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
    src/ol_sg_lasso_fp32.cpp
    src/ol_sg_lasso_fp64.cpp
    src/ol_sg_lasso_leastr_fp32.cpp
    src/ol_sg_lasso_leastr_fp64.cpp
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

# RPATH: look for shared libraries next to the binary so bundled
# libopenblas / libgfortran / etc. are found without LD_LIBRARY_PATH.
if(UNIX AND NOT APPLE)
    set_target_properties(myesl2 PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    # Use RPATH (not RUNPATH) so $ORIGIN resolves transitively for
    # indirect deps like libgfortran.so.5 (needed by libopenblas).
    target_link_options(myesl2 PRIVATE "LINKER:--disable-new-dtags")
    # Statically link libstdc++ and libgcc so binaries built with a newer
    # GCC do not require matching GLIBCXX_*/GCC_* symbols on the deploy
    # host (e.g. older HPC clusters). Glibc itself remains dynamic.
    target_link_options(myesl2 PRIVATE -static-libstdc++ -static-libgcc)
elseif(APPLE)
    set_target_properties(myesl2 PROPERTIES
        INSTALL_RPATH "@executable_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Link pre-compiled OpenBLAS and LAPACK from deps/ and copy runtime
# libraries (including the Fortran runtime) next to the binary.
if(WIN32)
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/windows-x64)
    target_link_libraries(myesl2 PRIVATE
        ${DEPS_DIR}/openblas.lib
        ${DEPS_DIR}/lapack.lib
    )
    # Core DLLs
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/openblas.dll
            ${DEPS_DIR}/lapack.dll
            $<TARGET_FILE_DIR:myesl2>
    )
    # Fortran runtime DLLs (required by lapack.dll)
    foreach(_dll libgfortran-5.dll libgcc_s_seh-1.dll)
        if(EXISTS "${DEPS_DIR}/${_dll}")
            add_custom_command(TARGET myesl2 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DEPS_DIR}/${_dll}" $<TARGET_FILE_DIR:myesl2>
            )
        endif()
    endforeach()
elseif(APPLE)
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/macos-arm64)
    target_link_libraries(myesl2 PRIVATE ${DEPS_DIR}/libopenblas.dylib)
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/libopenblas.dylib
            $<TARGET_FILE_DIR:myesl2>
    )
    # Fortran runtime dylibs (required by libopenblas)
    foreach(_dylib libgfortran.5.dylib libquadmath.0.dylib)
        if(EXISTS "${DEPS_DIR}/${_dylib}")
            add_custom_command(TARGET myesl2 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DEPS_DIR}/${_dylib}" $<TARGET_FILE_DIR:myesl2>
            )
        endif()
    endforeach()
else()
    set(DEPS_DIR ${PROJECT_SOURCE_DIR}/deps/linux-x64)
    # std::thread / pthread_create must be on the link line explicitly:
    # the manylinux2014 toolchain uses --no-copy-dt-needed-entries, so it
    # won't auto-pull libpthread in just because libopenblas.so depends
    # on it. The Ubuntu runner toolchain happens to be permissive enough
    # to let it through, but linking Threads::Threads is correct on both.
    find_package(Threads REQUIRED)
    target_link_libraries(myesl2 PRIVATE
        ${DEPS_DIR}/libopenblas.so
        Threads::Threads
    )
    # Copy both libopenblas.so and libopenblas.so.0 (the SONAME the linker
    # actually looks for at runtime) so $ORIGIN resolution works.
    add_custom_command(TARGET myesl2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/libopenblas.so $<TARGET_FILE_DIR:myesl2>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPS_DIR}/libopenblas.so.0 $<TARGET_FILE_DIR:myesl2>
    )
    # Fortran runtime (required by libopenblas)
    if(EXISTS "${DEPS_DIR}/libgfortran.so.5")
        add_custom_command(TARGET myesl2 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DEPS_DIR}/libgfortran.so.5" $<TARGET_FILE_DIR:myesl2>
        )
    endif()
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(myesl2 PRIVATE /W4)
else()
    target_compile_options(myesl2 PRIVATE -Wall -Wextra -Wpedantic)
endif()
