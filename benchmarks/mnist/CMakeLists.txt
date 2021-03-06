project(MNISTBenchmark)

find_package(Qt4)
find_package(OpenGL)
find_package(GLUT)

file(COPY benchmark.py DESTINATION .)

check_include_files("endian.h" HAVE_ENDIAN_H)

if(HAVE_ENDIAN_H)

  add_definitions("${OPENANN_COMPILER_FLAGS}")
  add_executable(MNIST mnist.cpp)
  target_link_libraries(MNIST openann)

  if(QT_FOUND AND GLUT_FOUND AND NOT CYGWIN_FOUND)
    set(QT_USE_QTMAIN TRUE)
    set(QT_USE_QTOPENGL TRUE)
    include(${QT_USE_FILE})
    add_definitions("${OPENANN_COMPILER_FLAGS} ${QT_DEFINITIONS} -DUSE_QT")
    include_directories(${QT_LIB_DIR})
    add_executable(Visualize visualize.cpp)
    target_link_libraries(Visualize openann ${QT_LIBRARIES} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
    add_executable(ViewData viewdata.cpp)
    target_link_libraries(ViewData openann ${QT_LIBRARIES} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
    add_executable(RBM rbm.cpp)
    target_link_libraries(RBM openann ${QT_LIBRARIES} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
  else()
    if(NOT QT_FOUND)
      message(WARNING "Qt4 is missing, cannot build MNIST visualization")
    endif()
    if(NOT GLUT_FOUND)
      message(WARNING "GLUT is missing, cannot build MNIST visualization")
    endif()
  endif()

else()

  message(STATUS "Could not find header 'endian.h'. Cannot compile mnist example.")

endif()
