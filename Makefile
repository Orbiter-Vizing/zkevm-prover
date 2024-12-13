TARGET_ZKP := zkProver
TARGET_BCT := bctree
TARGET_MNG := mainGenerator
TARGET_PLG := polsGenerator
TARGET_PLD := polsDiff
TARGET_TEST := zkProverTest
TARGET_W2DB := witness2db
TARGET_EXPRESSIONS := zkProverExpressions
TARGET_SETUP := fflonkSetup

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
CFLAGS := -Xcompiler -fopenmp
ASFLAGS := -felf64

CUDA_ARCH :=-arch=native

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D DEBUG
else
      CXXFLAGS += -O3  -D__USE_CUDA__ -DENABLE_EXPERIMENTAL_CODE
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

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./src/fflonk_setup/fflonk_setup*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/expressions/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" ! -path "./src/witness2db/*" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu \))
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_BCT := $(shell find ./tools/starkpil/bctree/build_const_tree.cpp ./tools/starkpil/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/poseidon_opt/poseidon_opt.cpp ./src/XKCP ./src/ffiasm ./src/starkpil/stark_info.* ./src/utils/* \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu \))
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find ./test/examples/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/merkleTreeGL.* ./src/starkpil/transcript/transcript.* ./src/starkpil/fri ./src/ffiasm ./src/utils ./tools/sm/sha256/sha256.cpp ./tools/sm/sha256/bcon/bcon_sha256.cpp ! -path "./src/starkpil/fri/friProveC12.*" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

SRCS_W2DB := ./src/witness2db/witness2db.cpp  ./src/goldilocks/src/goldilocks_base_field.cpp ./src/goldilocks/src/poseidon_goldilocks.cpp
OBJS_W2DB := $(SRCS_W2DB:%=$(BUILD_DIR)/%.o)
DEPS_W2DB := $(OBJS_W2DB:.o=.d)

SRCS_EXPRESSIONS := $(shell find ./test/expressions/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.*  ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.*  ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/ffiasm ./src/utils ! -path "./src/starkpil/fri/friProveC12.*"  \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))
OBJS_EXPRESSIONS := $(SRCS_EXPRESSIONS:%=$(BUILD_DIR)/%.o)
DEPS_EXPRESSIONS := $(OBJS_EXPRESSIONS:.o=.d)

SRCS_SETUP := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_SETUP += $(shell find src/XKCP -name *.cpp)
SRCS_SETUP += $(shell find src/fflonk_setup -name fflonk_setup.cpp)
SRCS_SETUP += $(shell find src/ffiasm/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_SETUP := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_SETUP))
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
DEPS_SETUP := $(OBJS_SETUP:.o=.d)

GOLDILOCKS_DIR := ./depends/goldilocks-gpu
GOLDILOCKS_LIBS := -L$(GOLDILOCKS_DIR) -lgl
GOLDILOCKS_INCLUDE_DIRS := -I./depends/goldilocks-gpu/include -I./depends/goldilocks-gpu/utils

ICICLE_DIR := ./depends/icicle/icicle
ICICLE_BUILD_DIR := $(BUILD_DIR)/icicle
ICICLE_LIBS := -L$(ICICLE_BUILD_DIR)/lib -lingo_curve_bn254 -lingo_field_bn254
ICICLE_INCLUDE_DIRS := -I$(ICICLE_DIR)/include

CPPFLAGS ?= $(INC_FLAGS) $(GOLDILOCKS_INCLUDE_DIRS) -MMD -MP

all: $(BUILD_DIR)/$(TARGET_ZKP)

goldilocks:
	make -C $(GOLDILOCKS_DIR) libgl

icicle:
	$(MKDIR_P) $(ICICLE_BUILD_DIR)
	cmake -S $(ICICLE_DIR) -B $(ICICLE_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254
	cmake --build $(ICICLE_BUILD_DIR)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)

expressions: ${BUILD_DIR}/$(TARGET_EXPRESSIONS)

$(BUILD_DIR)/$(TARGET_ZKP): goldilocks icicle $(OBJS_ZKP)
	$(CXX) $(CUDA_ARCH) $(OBJS_ZKP) -o $@ $(LDFLAGS) $(ICICLE_LIBS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(ICICLE_INCLUDE_DIRS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(CUDA_ARCH) $(OBJS_BCT) -o $@ $(LDFLAGS) $(ICICLE_LIBS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(ICICLE_INCLUDE_DIRS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(CUDA_ARCH) $(OBJS_TEST)  -o $@ $(LDFLAGS) $(ICICLE_LIBS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(ICICLE_INCLUDE_DIRS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

$(BUILD_DIR)/$(TARGET_EXPRESSIONS): $(OBJS_EXPRESSIONS)
	$(CXX) $(CUDA_ARCH) $(OBJS_EXPRESSIONS) -o $@ $(LDFLAGS) $(ICICLE_LIBS) $(GOLDILOCKS_LIBS) $(CFLAGS) $(CPPFLAGS) $(ICICLE_INCLUDE_DIRS) $(GOLDILOCKS_INCLUDE_DIRS) $(CXXFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CUDA_ARCH) --split-compile 0 $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(ICICLE_INCLUDE_DIRS) -c $< -o $@

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

witness2db: $(BUILD_DIR)/$(TARGET_W2DB)

$(BUILD_DIR)/$(TARGET_W2DB): $(OBJS_W2DB)
	$(CXX) $(OBJS_W2DB) $(CXXFLAGS_W2DB) -o $@ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS_W2DB) $(LDFLAGS_W2DB)

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(LDFLAGS) -o $@

.PHONY: clean

clean:
	$(RM) -rf $(BUILD_DIR)

-include $(DEPS_ZKP)
-include $(DEPS_SETUP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p
