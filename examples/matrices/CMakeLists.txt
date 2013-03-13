project(Matrices)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(Matrices matrices.cpp)
target_link_libraries(Matrices openann)
