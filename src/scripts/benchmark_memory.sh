# Memory measurement on SMALL SAMPLE ONLY (100 regexes max to keep profiling fast)
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/ce/polyglot/full
mkdir -p outputs/bva/polyglot/inner
mkdir -p logs

POLYGLOT_RANDOM_STRINGS_DIR="generated/ambiguous/polyglot-random-strings-100"
POLYGLOT_REGEX_FILE="polyglot/c-patterns-filtered-ambiguous-100-sampled.txt"

python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE \
	--log-file outputs/ce/polyglot/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SuperConfig" \
	--counter-type "counting-set" > logs/memory-polyglot-ce-full.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE \
	--log-file outputs/csa/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/memory-polyglot-csa-inner.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE \
	--log-file outputs/bva/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" > logs/memory-polyglot-bva-inner.txt 2>&1 &

wait


# Do the same for SNORT3 (all 125 regexes since it's small)
mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/ce/snort3/full
mkdir -p outputs/bva/snort3/inner
mkdir -p logs

SNORT3_RANDOM_STRINGS_DIR="generated/ambiguous/snort3-random-strings"
SNORT3_REGEX_FILE="snort3/c-patterns-filtered-ambiguous.txt"
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/ce/snort3/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SuperConfig" \
	--counter-type "counting-set" > logs/memory-snort3-ce-full.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/csa/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/memory-snort3-csa-inner.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/bva/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" > logs/memory-snort3-bva-inner.txt 2>&1 &
wait