#!/bin/bash
# SYCL vs TernOCL, int2 upcvt: best tile of each (same tile list) per shape,
# then the two winners re-measured alternately REPS times, median reported.
# >= 2 GiB rotating distinct weights, device-event times. The sweep alternates
# SYCL and OCL per tile. PACE=s (LNL) sleeps before each re-measure, and the
# TOPK (default 4) fastest sweep tiles of each are re-timed paced before picking.
# PIN="taskset -c N" prefixes both drivers (run_all.sh sets an E-core on LNL).
#   DT=fp16|bf16 M=1 SHAPES="name:K:N ..." bash bench.sh <out.txt>
# TERNOCL = TernOCL checkout with built drivers (default ../TernOCL).
HERE=$(cd "$(dirname "$0")" && pwd)
TERNOCL=${TERNOCL:-$HERE/../TernOCL}
DT=${DT:-fp16}; OUT=$1
S="$PIN $HERE/build/int2_fp16_upcvt_sycl --dtype $DT"
O="$PIN $TERNOCL/int2_fp16_upcvt/build/int2_fp16_upcvt_ocl --dtype $DT"
M=${M:-1}; IT=${IT:-20}; REPS=${REPS:-3}
TOPK=${TOPK:-$([[ ${PACE:-0} == 0 ]] && echo 1 || echo 4)}
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
SHAPES=${SHAPES:-"8B.qkv:4096:6144 8B.o_proj:4096:4096 8B.gate_up:4096:24576 8B.down:12288:4096 8B.lm_head:4096:151680 27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
if [[ $M -gt 1 ]]; then
  T=""; for t in "32 16" "64 16" "96 16" "128 16" "32 32" "64 32"; do for wg in "2 4" "4 4" "2 2" "1 8" "4 2" "8 1"; do
    set -- $t $wg; T+="--mt-m $1 --mt-n $2 --wg-m $3 --wg-n $4|"; done; done
else
  T=""; for wn in 16 32 64; do for ls in 1 2 4 6 8; do for u in 1 2; do T+="--wgn $wn --ls $ls --u $u|"; done; done; done
fi
T=${T%|}
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
run() { $1 --m $M --k $K --n $N $2 --iters $IT --no-validate 2>&1 | dt; }
sweep() {  # -> lines "impl time cfg"
  local c t impl bin; IFS='|' read -ra cs <<<"$T"
  for c in "${cs[@]}"; do for impl in sycl ocl; do
    bin=$S; [[ $impl == ocl ]] && bin=$O
    t=$(run "$bin" "$c"); echo "$name $impl [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && echo "$impl $t $c"
  done; done
}
pick() {  # pick <impl> <sweep lines> -> cfg
  local bin=$S bc="" bt=1e9 c t top; [[ $1 == ocl ]] && bin=$O
  mapfile -t top < <(grep "^$1 " <<<"$2" | sort -k2 -g | head -$TOPK | cut -d' ' -f3-)
  [[ ${#top[@]} -eq 1 ]] && { echo "${top[0]}"; return; }
  for c in "${top[@]}"; do
    sleep ${PACE:-0}; t=$(run "$bin" "$c"); echo "$name $1 paced [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc"
}
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  sw=$(sweep); sc=$(pick sycl "$sw"); oc=$(pick ocl "$sw")
  ss=(); os=()
  for r in $(seq $REPS); do
    sleep ${PACE:-0}
    os+=($($O --m $M --k $K --n $N $oc --iters $IT --no-validate 2>&1 | dt))
    sleep ${PACE:-0}
    ss+=($($S --m $M --k $K --n $N $sc --iters $IT --no-validate 2>&1 | dt))
  done
  to=$(med "${os[@]}"); ts=$(med "${ss[@]}")
  printf "%-12s %s M=%-4s K=%-5s N=%-6s | ocl(%s) %s ms | sycl(%s) %s ms | sycl/ocl speed x%.3f\n" \
    $name $DT $M $K $N "$oc" $to "$sc" $ts $(awk "BEGIN{print $to/$ts}") | tee -a $OUT
  echo "   reps ocl: ${os[*]} | sycl: ${ss[*]}" | tee -a $OUT
done
