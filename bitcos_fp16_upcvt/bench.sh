#!/bin/bash
# SYCL vs TernOCL, BITCOS: best tile of each (same tile list) per shape, then
# the two winners re-measured alternately REPS times, median reported.
# >= 2 GiB rotating distinct weights, device-event times.
# PACE=s sleeps before each re-measure (LNL).
#   DT=fp16|bf16 M=1 Z=0.40 SHAPES="name:K:N ..." bash bench.sh <out.txt>
# TERNOCL = TernOCL checkout with built drivers (default ../TernOCL).
HERE=$(cd "$(dirname "$0")" && pwd)
TERNOCL=${TERNOCL:-$HERE/../TernOCL}
DT=${DT:-fp16}; OUT=$1
S="$HERE/build/bitcos_fp16_upcvt_sycl --dtype $DT"
O="$TERNOCL/bitcos_fp16_upcvt/build/bitcos_fp16_upcvt_ocl --dtype $DT"
M=${M:-1}; Z=${Z:-0.40}; IT=${IT:-20}; REPS=${REPS:-3}
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
SHAPES=${SHAPES:-"27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384 27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320"}
dt() { grep -a "Avg dev   time" | sed 's/.*: \([0-9.]*\) ms.*/\1/'; }
if [[ $M -gt 8 ]]; then
  T=""; for t in "16 16" "32 16" "64 16" "16 32" "32 32" "16 64"; do for wg in "1 4" "1 8" "2 2" "2 4" "4 2"; do
    set -- $t $wg; T+="--mt-m $1 --mt-n $2 --wg-m $3 --wg-n $4|"; done; done
else
  T=""; for c in 32:1 32:2 32:4 32:8 32:16 64:1 64:2 64:4 64:8 64:16 128:1 128:2 128:4 128:8 256:1 256:2 256:4; do
    T+="--wgn ${c%:*} --ls ${c#*:}|"; done
fi
T=${T%|}
med() { printf "%s\n" "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
best() {  # best <binary> <label> -> "cfg|time"
  local bin=$1 bc="" bt=1e9 c t; IFS='|' read -ra cs <<<"$T"
  for c in "${cs[@]}"; do
    t=$($bin --m $M --k $K --n $N --z $Z $c --iters $IT --no-validate 2>&1 | dt)
    echo "$name $2 [$c] $t" >> $OUT.sweep
    [[ -n "$t" ]] && awk "BEGIN{exit !($t < $bt)}" && { bt=$t; bc=$c; }
  done
  echo "$bc|$bt"
}
for sh in $SHAPES; do IFS=: read name K N <<<"$sh"
  IFS='|' read sc _ <<<"$(best "$S" sycl)"
  IFS='|' read oc _ <<<"$(best "$O" ocl)"
  ss=(); os=()
  for r in $(seq $REPS); do
    sleep ${PACE:-0}
    os+=($($O --m $M --k $K --n $N --z $Z $oc --iters $IT --no-validate 2>&1 | dt))
    sleep ${PACE:-0}
    ss+=($($S --m $M --k $K --n $N --z $Z $sc --iters $IT --no-validate 2>&1 | dt))
  done
  to=$(med "${os[@]}"); ts=$(med "${ss[@]}")
  printf "%-12s %s z=%s M=%-4s K=%-5s N=%-6s | ocl(%s) %s ms | sycl(%s) %s ms | sycl/ocl speed x%.3f\n" \
    $name $DT $Z $M $K $N "$oc" $to "$sc" $ts $(awk "BEGIN{print $to/$ts}") | tee -a $OUT
  echo "   reps ocl: ${os[*]} | sycl: ${ss[*]}" | tee -a $OUT
done
