# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lixj/Downloads/trt_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lixj/Downloads/trt_test/build

# Include any dependencies generated for this target.
include CMakeFiles/convert_model.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/convert_model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convert_model.dir/flags.make

CMakeFiles/convert_model.dir/src/convert_model.cpp.o: CMakeFiles/convert_model.dir/flags.make
CMakeFiles/convert_model.dir/src/convert_model.cpp.o: ../src/convert_model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lixj/Downloads/trt_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/convert_model.dir/src/convert_model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convert_model.dir/src/convert_model.cpp.o -c /home/lixj/Downloads/trt_test/src/convert_model.cpp

CMakeFiles/convert_model.dir/src/convert_model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_model.dir/src/convert_model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lixj/Downloads/trt_test/src/convert_model.cpp > CMakeFiles/convert_model.dir/src/convert_model.cpp.i

CMakeFiles/convert_model.dir/src/convert_model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_model.dir/src/convert_model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lixj/Downloads/trt_test/src/convert_model.cpp -o CMakeFiles/convert_model.dir/src/convert_model.cpp.s

# Object files for target convert_model
convert_model_OBJECTS = \
"CMakeFiles/convert_model.dir/src/convert_model.cpp.o"

# External object files for target convert_model
convert_model_EXTERNAL_OBJECTS =

libconvert_model.a: CMakeFiles/convert_model.dir/src/convert_model.cpp.o
libconvert_model.a: CMakeFiles/convert_model.dir/build.make
libconvert_model.a: CMakeFiles/convert_model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lixj/Downloads/trt_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libconvert_model.a"
	$(CMAKE_COMMAND) -P CMakeFiles/convert_model.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convert_model.dir/build: libconvert_model.a

.PHONY : CMakeFiles/convert_model.dir/build

CMakeFiles/convert_model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convert_model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convert_model.dir/clean

CMakeFiles/convert_model.dir/depend:
	cd /home/lixj/Downloads/trt_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lixj/Downloads/trt_test /home/lixj/Downloads/trt_test /home/lixj/Downloads/trt_test/build /home/lixj/Downloads/trt_test/build /home/lixj/Downloads/trt_test/build/CMakeFiles/convert_model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/convert_model.dir/depend
