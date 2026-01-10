# Benchmark inner vs outer expansion for counting-set approach

mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/csa/polyglot/outer
mkdir -p logs

POLYGLOT_RANDOM_STRINGS_DIR="generated/nested-counters/polyglot-random-strings"
POLYGLOT_REGEX_FILE="polyglot/nested-c-patterns-filtered.txt"

# TIMING TESTS on FULL nested dataset
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE \
	--timing-log-file outputs/csa/polyglot/outer/nested-random-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/expansion-polyglot-csa-outer.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE \
	--timing-log-file outputs/csa/polyglot/inner/nested-random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/expansion-polyglot-csa-inner.txt 2>&1 &

# MEMORY MEASUREMENT on SMALL SAMPLE ONLY (100 regexes max to keep profiling fast)
# Sample first 100 nested regexes  
head -100 polyglot/nested-c-patterns-filtered.txt > /tmp/nested-sample-100.txt


python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file /tmp/nested-sample-100.txt \
	--log-file outputs/csa/polyglot/outer/nested-peak-mem-usage.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig" > logs/expansion-polyglot-csa-outer-memory.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_avg_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file /tmp/nested-sample-100.txt \
	--log-file outputs/csa/polyglot/inner/nested-peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig" > logs/expansion-polyglot-csa-inner-memory.txt 2>&1 &
wait
rm -f /tmp/nested-sample-100.txt