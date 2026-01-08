# Benchmark inner vs outer expansion for counting-set approach

# TIMING TESTS on FULL nested dataset
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/outer/random-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
wait

# MEMORY MEASUREMENT on SMALL SAMPLE ONLY (100 regexes max to keep profiling fast)
# Sample first 100 nested regexes  
head -100 polyglot/nested-c-patterns-filtered.txt > /tmp/nested-sample-100.txt

python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file /tmp/nested-sample-100.txt \
	--log-file outputs/csa/polyglot/outer/nested-peak-mem-usage.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig" &
python -OO -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file /tmp/nested-sample-100.txt \
	--log-file outputs/csa/polyglot/inner/nested-peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig" &
wait
rm -f /tmp/nested-sample-100.txt