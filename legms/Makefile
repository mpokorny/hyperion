# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

PROJDIR := $(realpath $(CURDIR))

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
USE_LLVM        ?= 0		# Include LLVM support (requires llvm)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= liblegms.so

# List all the application source files here
GEN_SRC		?= tree_index_space.cc \
	utility.cc \
	Grids_c.cc \
	ms/Column.cc \
	ms/Column_c.cc \
	ms/ColumnPartition_c.cc \
	ms/MSTable.cc \
	ms/MSTable_c.cc \
	ms/Table.cc \
	ms/Table_c.cc \
	ms/TableReadTask.cc \
	ms/TableReadTask_c.cc

GEN_GPU_SRC	?=				# .cu files

# You can modify these variables, some will be appended to by the runtime
# makefile
CASACORE	?= /users/mpokorny/projects/casacore.git/casacore-install
REGENT	?= $(LG_RT_DIR)/../language/build

CC_FLAGS	?= -std=c++17 -Wall -Werror
NVCC_FLAGS	?=
GASNET_FLAGS	?=
INC_FLAGS	?= -I$(CASACORE)/include -I$(PROJDIR) -I$(PROJDIR)/ms
LD_FLAGS	?= -lstdc++fs \
	-L$(CASACORE)/lib -lcasa_tables -lcasa_casa \
	-Wl,-rpath -Wl,$(CASACORE)/lib \
	-L$(REGENT)/lib -llegion -lrealm \
	-Wl,-rpath -Wl,$(REGENT)/lib

CC_FLAGS += -fPIC

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
else
	LD_FLAGS += -shared
endif

# ifeq ($(shell uname), Darwin)
# 	LD_FLAGS += -Wl,-force_load,liblegion.a -Wl,-force_load,librealm.a
# else
# 	LD_FLAGS += -Wl,--whole-archive -llegion -lrealm -Wl,--no-whole-archive
# endif

NO_BUILD_RULES := 1
GEN_OBJS	:= $(GEN_SRC:.cc=.cc.o)

.PHONY: all
all: $(OUTFILE)

$(GEN_OBJS) : %.cc.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(OUTFILE) : $(GEN_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(CXX) -o $(OUTFILE) $(GEN_OBJS) $(LD_FLAGS)

clean::
	$(RM) -f $(OUTFILE) $(GEN_OBJS)

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk
