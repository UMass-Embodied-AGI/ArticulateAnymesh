
##############################
### Find required packages ###
##############################
find_package(PythonLibs)
find_package(PythonInterp)
include(FindNumPy)
include_directories(../include)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
if (${Boost_MINOR_VERSION} LESS 67 )
    find_package(Boost 1.65.0 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
else()
    set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    find_package(Boost 1.67.0 COMPONENTS numpy${PYTHONVERSION} REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})



message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

file(GLOB CP_HEADERS include/*.h)

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11")
message(${Boost_LIBRARIES})
add_library(cp SHARED cutpursuit.cpp ${CP_HEADERS})
target_link_libraries(cp
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES})