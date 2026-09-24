# SYCL Hadamard FWHT input transform (+ validating driver).
#   source /swtools/intel/2026.0/oneapi-vars.sh --force
#   make && ./build/hadamard_sycl --rows 1024 --k 5120
CXX       = icpx
AOT_DEVICES ?= bmg-g31,lnl-m
CXXFLAGS  = -fsycl -std=c++20 -O3 -fp-model=precise -Wno-psabi -I../common
LDFLAGS   = -fsycl
ifndef JIT
CXXFLAGS += -fsycl-targets=spir64_gen
LDFLAGS  += -fsycl-targets=spir64_gen -Xs "-device $(AOT_DEVICES)"
endif

BUILD  = build
TARGET = $(BUILD)/hadamard_sycl

all: $(TARGET)

$(TARGET): main.cpp hadamard_fwht.hpp ../common/xe2.hpp ../common/dt16.hpp ../common/driver.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
