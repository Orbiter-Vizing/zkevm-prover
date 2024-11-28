TARGET_ZKP := zkProver
TARGET_BCT := bctree
TARGET_MNG := mainGenerator
TARGET_PLG := polsGenerator
TARGET_PLD := polsDiff
TARGET_TEST := zkProverTest

BUILD_DIR := ./build
SRC_DIRS := ./src ./test ./tools
SETUP_DIRS := ./src/rapidsnark

GRPCPP_FLAGS := $(shell pkg-config grpc++ --cflags)
GRPCPP_LIBS := $(shell pkg-config grpc++ --libs) -lgrpc++_reflection
ifndef GRPCPP_LIBS
$(error gRPC++ could not be found via pkg-config, you need to install them)
endif

CXX := nvcc
AS := nasm
CXXFLAGS := -std=c++17 --expt-relaxed-constexpr -Xcompiler "-Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 $(GRPCPP_FLAGS)" #-Wfatal-errors
LDFLAGS := -lprotobuf -lsodium -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lgmpxx -lsecp256k1 -lcrypto -luuid -liomp5 -lssl $(GRPCPP_LIBS)
CXXFLAGS_W2DB := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2
LDFLAGS_W2DB := -lgmp -lstdc++ -lgmpxx
CFLAGS := -Xcompiler -fopenmp
ASFLAGS := -felf64

# only for RTX4090
CUDA_ARCH := -gencode arch=compute_89,code=sm_89
# CUDA_ARCH :=-arch=native

use_cuda ?= 1

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D DEBUG
else
      CXXFLAGS += -O3
	  ifeq ($(use_cuda),1)
	  CXXFLAGS += -D__USE_CUDA__
	  endif
endif

#ifneq ($(avx512),0)
#ifeq ($(avx512),1)
#	CXXFLAGS += -mavx512f -D__AVX512__
#else
## check if AVX-512 is supported
#AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)
#ifneq ($(AVX512_SUPPORTED),)
#	CXXFLAGS += -mavx512f -D__AVX512__
#endif
#endif
#endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./src/test/*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/prover/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_BCT := $(shell find $(SRC_DIRS) ! -path "./src/test/*" ! -path "./src/main.cpp" ! -path "./test/prover/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find $(SRC_DIRS) ! -path "./src/test/*" ! -path "./src/main.cpp" ! -path "./tools/starkpil/bctree/*"  ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

ICICLE_DIR := ./depends/icicle/icicle
ICICLE_BUILD_DIR := $(BUILD_DIR)/icicle
ICICLE_LIB_DIRS := -L$(ICICLE_BUILD_DIR)/lib
ICICLE_LIBS := -lingo_curve_bn254 -lingo_field_bn254
ICICLE_INCLUDE_DIRS := -I./depends/icicle/icicle/include

GOLDILOCKS_DIR := src/goldilocks
GOLDILOCKS_LIBS := -L$(BUILD_DIR) -lgl
GOLDILOCKS_INCLUDE_DIRS := -I./src/goldilocks/include -I./src/goldilocks/utils

CPPFLAGS ?= $(INC_FLAGS) $(GOLDILOCKS_INCLUDE_DIRS) -MMD -MP

all: $(BUILD_DIR)/$(TARGET_ZKP)

goldilocks:
	make -C $(GOLDILOCKS_DIR) libgl
	mv $(GOLDILOCKS_DIR)/libgl.a $(BUILD_DIR)/

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)

$(BUILD_DIR)/$(TARGET_ZKP): goldilocks $(OBJS_ZKP)
	$(CXX) $(CUDA_ARCH) $(OBJS_ZKP) -o $@ $(LDFLAGS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(CUDA_ARCH) $(OBJS_BCT) -o $@ $(LDFLAGS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(CUDA_ARCH) $(OBJS_TEST) -o $@ $(LDFLAGS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CUDA_ARCH) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CUDA_ARCH) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CUDA_ARCH) --split-compile 0 $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

main_generator: $(BUILD_DIR)/$(TARGET_MNG)

$(BUILD_DIR)/$(TARGET_MNG): ./src/main_generator/main_generator.cpp ./src/config/definitions.hpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/main_generator/main_generator.cpp -o $@ -lgmp

pols_generator: $(BUILD_DIR)/$(TARGET_PLG)

$(BUILD_DIR)/$(TARGET_PLG): ./src/pols_generator/pols_generator.cpp ./src/config/definitions.hpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_generator/pols_generator.cpp -o $@ -lgmp

pols_diff: $(BUILD_DIR)/$(TARGET_PLD)

$(BUILD_DIR)/$(TARGET_PLD): ./src/pols_diff/pols_diff.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_diff/pols_diff.cpp ./src/config/fork_info.* $(CXXFLAGS) $(INC_FLAGS) -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	$(RM) -rf $(BUILD_DIR)

-include $(DEPS_ZKP)
-include $(DEPS_SETUP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p
