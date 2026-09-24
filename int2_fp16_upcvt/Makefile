# SYCL int2 x fp16/bf16 upcvt GEMM/GEMV driver.
#   source /swtools/intel/2026.0/oneapi-vars.sh --force
#   make && ./build/int2_fp16_upcvt_sycl --m 1 --n 17408 --k 5120
CXX       = icpx
# AOT for the B70 (BMG-G31) and Lunar Lake; JIT=1 builds SPIR-V only (JIT at run time,
# measurably slower for some kernels, see README)
AOT_DEVICES ?= bmg-g31,lnl-m
CXXFLAGS  = -fsycl -std=c++20 -O3 -fp-model=precise -qopenmp -Wno-psabi -I../common \
            -fsycl-device-code-split=per_kernel
LDFLAGS   = -fsycl -qopenmp
ifndef JIT
CXXFLAGS += -fsycl-targets=spir64_gen
LDFLAGS  += -fsycl-targets=spir64_gen -Xs "-device $(AOT_DEVICES)" -fsycl-max-parallel-link-jobs=16
endif

BUILD  = build
TARGET = $(BUILD)/int2_fp16_upcvt_sycl
DEPS   = int2_fp16_upcvt.hpp ../common/xe2.hpp ../common/epilogue_dev.hpp ../common/epilogue.hpp \
         ../common/dt16.hpp ../common/driver.hpp

all: $(TARGET)

$(TARGET): main.cpp $(DEPS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
