#!/bin/bash
# SYCL vs TernOCL, int2 x int8 DPAS: per qmode, best tile of each (same tile
# list) per shape, then the winners re-measured alternately REPS times, median
# reported. Times include the A pre-kernel. >= 2 GiB rotating weights.
# PACE=s sleeps before each re-measure (LNL).
#   DT=fp16|bf16 M=1 QMODES="0 1" SHAPES="name:K:N ..." bash bench.sh <out.txt>
# TERNOCL = TernOCL checkout with built drivers (default ../TernOCL).
HERE=$(cd "$(dirname "$0")" && pwd)
TERNOCL=${TERNOCL:-$HERE/../TernOCL}
DT=${DT:-fp16}; OUT=$1
S="$HERE/build/int2_int8_dpas_sycl --dtype $DT"
O="$TERNOCL/int2_via_int2_x_int8_dpas/build/int2_int8_dpas_ocl --dtype $DT"
M=${M:-1}; IT=${IT:-20}; REPS=${REPS:-3}; QMODES=${QMODES:-0 1}
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
SHAPES=${SHAPES:-"8B.qkv:4096:6144 8B.o_proj:4096:4096 8B.gate_up:4096:24576 8B.down:12288:4096 8B.lm_head:4096:151680 27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
if [[ $M -gt 1 ]]; then
  T="--mt-m 8 --mt-n 128 --wg-m 8 --wg-n 2|--mt-m 8 --mt-n 128 --wg-m 4 --wg-n 2|--mt-m 8 --mt-n 128 --wg-m 4 --wg-n 4|--mt-m 8 --mt-n 128 --wg-m 16 --wg-n 1|--mt-m 8 --mt-n 128 --wg-m 2 --wg-n 4|--mt-m 16 --mt-n 64 --wg-m 4 --wg-n 2|--mt-m 16 --mt-n 64 --wg-m 8 --wg-n 2|--mt-m 8 --mt-n 64 --wg-m 8 --wg-n 2|--mt-m 32 --mt-n 32 --wg-m 4 --wg-n 2"
else
  T=""; for nsg in 1 2 4; do for ls in 1 2 4 8; do for u in 1 2; do T+="--nsg $nsg --ls $ls --u $u|"; done; done; done; T=${T%|}
fi
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
best() {  # best <binary> <label> -> "cfg|time"
  local bin=$1 bc="" bt=1e9 c t; IFS='|' read -ra cs <<<"$T"
  for c in "${cs[@]}"; do
    t=$($bin --m $M --k $K --n $N --qmode $qm $c --iters $IT --no-validate 2>&1 | dt)
    echo "$name $2 qm=$qm [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc|$bt"
}
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  for qm in $QMODES; do
    IFS='|' read sc _ <<<"$(best "$S" sycl)"
    IFS='|' read oc _ <<<"$(best "$O" ocl)"
    ss=(); os=()
    for r in $(seq $REPS); do
      sleep ${PACE:-0}
      os+=($($O --m $M --k $K --n $N --qmode $qm $oc --iters $IT --no-validate 2>&1 | dt))
      sleep ${PACE:-0}
      ss+=($($S --m $M --k $K --n $N --qmode $qm $sc --iters $IT --no-validate 2>&1 | dt))
    done
    to=$(med "${os[@]}"); ts=$(med "${ss[@]}")
    printf "%-12s %s qm=%s M=%-4s K=%-5s N=%-6s | ocl(%s) %s ms | sycl(%s) %s ms | sycl/ocl speed x%.3f\n" \
      $name $DT $qm $M $K $N "$oc" $to "$sc" $ts $(awk "BEGIN{print $to/$ts}") | tee -a $OUT
    echo "   reps ocl: ${os[*]} | sycl: ${ss[*]}" | tee -a $OUT
  done
done
