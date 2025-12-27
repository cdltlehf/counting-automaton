# Memory measurement on SMALL SAMPLE ONLY (100 regexes max to keep profiling fast)
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/ce/polyglot/full
mkdir -p outputs/bva/polyglot/inner

POLYGLOT_RANDOM_STRINGS_DIR="generated/counters/polyglot-random-strings"
POLYGLOT_REGEX_FILE="polyglot/c-patterns-filtered.txt"

# Sample first 100 regexes for memory profiling
head -100 $POLYGLOT_REGEX_FILE > /tmp/polyglot-sample-100.txt

python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file /tmp/polyglot-sample-100.txt \
	--log-file outputs/ce/polyglot/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SuperConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file /tmp/polyglot-sample-100.txt \
	--log-file outputs/csa/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file /tmp/polyglot-sample-100.txt \
	--log-file outputs/bva/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" &

wait
rm -f /tmp/polyglot-sample-100.txt


# Do the same for SNORT3 (all 125 regexes since it's small)
mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/ce/snort3/full
mkdir -p outputs/bva/snort3/inner

SNORT3_RANDOM_STRINGS_DIR="generated/counters/snort3-random-strings"
SNORT3_REGEX_FILE="snort3/c-patterns-filtered.txt"
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/ce/snort3/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SuperConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/csa/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE \
	--log-file outputs/bva/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 3 \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" &
wait