#!/bin/bash
# Validate TernSYCL and benchmark it against TernOCL on the current GPU node.
#   bash run_all.sh [validate] [epilogues] [bench]
#   env: VARIANT=int2_fp16_upcvt|int2_via_int2_x_int8_dpas|all  DTYPES="fp16 bf16"  MS="1 1024"
#        TERNOCL=<TernOCL checkout with built drivers> (default ./TernOCL)
#        CARDS="0 1 ..." spread the bench shapes over these GPUs (ZE_AFFINITY_MASK)
#        PACE=s sleep before each re-measure (LNL: shared-memory bandwidth drifts)
#        PIN="cmd" host-process prefix for both drivers (LNL default: taskset -c 4, an E-core;
#        a driver spinning on a P-core takes package power from the GPU under PL1)
# Results go to <variant>/results/{val,bench}_<arch>_<dtype>_m<M>.txt (+ .sweep with
# every tile tried). The architecture (b70 / lnl) comes from the hostname; ARCH= overrides.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=${ARCH:-$(case $(hostname -s) in *lnl*) echo lnl;; *) echo b70;; esac)}
export PIN=${PIN-$([[ $ARCH == lnl ]] && echo "taskset -c 4")}
VARIANT=${VARIANT:-all}; DTYPES=${DTYPES:-fp16 bf16}; MS=${MS:-1 1024}; CARDS=${CARDS:-}
export TERNOCL=${TERNOCL:-$HERE/TernOCL}
[[ $VARIANT == all ]] && VARIANT="int2_fp16_upcvt int2_via_int2_x_int8_dpas"
DO=${*:-validate epilogues bench}
unset IGC_ShaderDumpEnable IGC_DumpToCustomDir
SHAPES_ALL=(8B.qkv:4096:6144 8B.o_proj:4096:4096 8B.gate_up:4096:24576 8B.down:12288:4096
            8B.lm_head:4096:151680 27B.gate_up:5120:34816 27B.down:17408:5120 27B.qkvz:5120:16384
            27B.out_proj:6144:5120 27B.qkv:5120:14336 27B.lm_head:5120:248320)
if [[ " $DO " == *" validate "* ]]; then
  mkdir -p $HERE/hadamard/results
  for dt in fp16 bf16; do for s in "--rows 1 --k 5120" "--rows 1024 --k 5120" "--rows 77 --k 17408 --no-signs" "--rows 33 --k 6144 --inverse"; do
    echo "hadamard $dt $s | $($HERE/hadamard/build/hadamard_sycl $s --dtype $dt 2>&1 | grep -a 'summary')"
  done; done > $HERE/hadamard/results/val_$ARCH.txt
  echo "hadamard validate [$ARCH]: $(grep -c PASSED $HERE/hadamard/results/val_$ARCH.txt)/8 passed"
fi
for v in $VARIANT; do
  d=$HERE/$v; mkdir -p $d/results
  if [[ " $DO " == *" validate "* ]]; then
    bash $d/validate.sh > $d/results/val_$ARCH.txt 2>&1
    echo "$v validate [$ARCH]: $(grep -c '2/2 sets PASSED' $d/results/val_$ARCH.txt) of" \
         "$(grep -c 'Validation summary' $d/results/val_$ARCH.txt) cases passed"
  fi
  if [[ " $DO " == *" epilogues "* ]]; then
    bash $HERE/validate_epilogues.sh $v > $d/results/val_epi_$ARCH.txt 2>&1 || true
    echo "$v epilogues [$ARCH]: $(tail -1 $d/results/val_epi_$ARCH.txt)"
  fi
  if [[ " $DO " == *" bench "* ]]; then
    for dt in $DTYPES; do for m in $MS; do
      out=$d/results/bench_${ARCH}_${dt}_m$m.txt; rm -f $out $out.sweep
      it=20; [[ $m -gt 1 ]] && it=10; [[ $v == int2_fp16_upcvt && $m == 1 ]] && it=50
      if [[ -z "$CARDS" ]]; then
        DT=$dt M=$m IT=$it SHAPES="${SHAPES_ALL[*]}" bash $d/bench.sh $out > /dev/null
      else
        cs=($CARDS); n=${#cs[@]}
        for ((c = 0; c < n; ++c)); do
          list=""; for ((i = c; i < ${#SHAPES_ALL[@]}; i += n)); do list+="${SHAPES_ALL[$i]} "; done
          [[ -z "$list" ]] && continue
          ZE_AFFINITY_MASK=${cs[$c]} DT=$dt M=$m IT=$it SHAPES="$list" bash $d/bench.sh $out.card$c > /dev/null &
        done
        wait
        for s in "${SHAPES_ALL[@]}"; do cat $out.card? | grep -a --no-group-separator -A1 -E "^${s%%:*} +(fp16|bf16) " >> $out || true; done
        cat $out.card*.sweep > $out.sweep; rm -f $out.card*
      fi
      echo "$v bench [$ARCH $dt M=$m] -> $out"
    done; done
  fi
done
