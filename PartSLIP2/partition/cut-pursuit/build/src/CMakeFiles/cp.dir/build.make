# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /work/pi_chuangg_umass_edu/xiaowen/env/partslip/bin/cmake

# The command to remove a file.
RM = /work/pi_chuangg_umass_edu/xiaowen/env/partslip/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build

# Include any dependencies generated for this target.
include src/CMakeFiles/cp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/cp.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cp.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cp.dir/flags.make

src/CMakeFiles/cp.dir/codegen:
.PHONY : src/CMakeFiles/cp.dir/codegen

src/CMakeFiles/cp.dir/cutpursuit.cpp.o: src/CMakeFiles/cp.dir/flags.make
src/CMakeFiles/cp.dir/cutpursuit.cpp.o: /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/src/cutpursuit.cpp
src/CMakeFiles/cp.dir/cutpursuit.cpp.o: src/CMakeFiles/cp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/cp.dir/cutpursuit.cpp.o"
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src && /modules/spack/packages/linux-ubuntu24.04-x86_64_v3/gcc-13.2.0/gcc-9.4.0-ctk4n3223pjnwfvyn6crmzo4isburta7/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/cp.dir/cutpursuit.cpp.o -MF CMakeFiles/cp.dir/cutpursuit.cpp.o.d -o CMakeFiles/cp.dir/cutpursuit.cpp.o -c /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/src/cutpursuit.cpp

src/CMakeFiles/cp.dir/cutpursuit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cp.dir/cutpursuit.cpp.i"
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src && /modules/spack/packages/linux-ubuntu24.04-x86_64_v3/gcc-13.2.0/gcc-9.4.0-ctk4n3223pjnwfvyn6crmzo4isburta7/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/src/cutpursuit.cpp > CMakeFiles/cp.dir/cutpursuit.cpp.i

src/CMakeFiles/cp.dir/cutpursuit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cp.dir/cutpursuit.cpp.s"
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src && /modules/spack/packages/linux-ubuntu24.04-x86_64_v3/gcc-13.2.0/gcc-9.4.0-ctk4n3223pjnwfvyn6crmzo4isburta7/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/src/cutpursuit.cpp -o CMakeFiles/cp.dir/cutpursuit.cpp.s

# Object files for target cp
cp_OBJECTS = \
"CMakeFiles/cp.dir/cutpursuit.cpp.o"

# External object files for target cp
cp_EXTERNAL_OBJECTS =

src/libcp.so: src/CMakeFiles/cp.dir/cutpursuit.cpp.o
src/libcp.so: src/CMakeFiles/cp.dir/build.make
src/libcp.so: src/CMakeFiles/cp.dir/compiler_depend.ts
src/libcp.so: /work/pi_chuangg_umass_edu/xiaowen/env/partslip/lib/libboost_numpy310.so.1.83.0
src/libcp.so: /work/pi_chuangg_umass_edu/xiaowen/env/partslip/lib/libpython3.10.so
src/libcp.so: /work/pi_chuangg_umass_edu/xiaowen/env/partslip/lib/libboost_python310.so.1.83.0
src/libcp.so: src/CMakeFiles/cp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcp.so"
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cp.dir/build: src/libcp.so
.PHONY : src/CMakeFiles/cp.dir/build

src/CMakeFiles/cp.dir/clean:
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cp.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cp.dir/clean

src/CMakeFiles/cp.dir/depend:
	cd /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/src /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src /work/pi_chuangg_umass_edu/xiaowen/PartSLIP2/partition/cut-pursuit/build/src/CMakeFiles/cp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/cp.dir/depend

