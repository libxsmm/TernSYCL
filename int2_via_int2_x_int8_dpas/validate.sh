#!/bin/bash
# Validation: SYCL int2 x int8 DPAS (both qmodes, GEMM + GEMV + ragged shapes, distinct sets).
#   bash validate.sh
cd "$(dirname "$0")"; S=./build/int2_int8_dpas_sycl
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
summ() { grep -a 'max ULP' | sed 's/.*max abs diff \([0-9.]*\).*max ULP diff \([0-9]*\).*pass rate \(.*\)/abs \1 ulp \2 pass \3/' | tr '\n' ' '; }
for dt in fp16 bf16; do for qm in 0 1; do
  for mkn in "1024 17408 5120" "1024 5120 14336" "1024 5120 34816" "1000 5120 4096" "77 128 48" \
             "129 256 272" "200 5120 16384" "1 5120 34816" "1 17408 5120" "1 5120 248320" "4 6144 5120"; do
    set -- $mkn
    o=$($S --m $1 --k $2 --n $3 --dtype $dt --qmode $qm --iters 3 --sets 2 --distinct-sets 2>&1)
    echo "sycl $dt qm=$qm M=$1 K=$2 N=$3 | $(grep -a 'scale_a' <<<"$o") | $(summ <<<"$o")| $(grep -a 'Validation summary\|error\|failed' <<<"$o" | head -2)"
  done
done; done
