# Benchmark inner vs outer expansion for counting-set approach
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/outer/random-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set"
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set"
# No need to repeat the inner expansion here since it was already done above

# Benchmark inner vs outer expansion for counting-set approach
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--log-file outputs/csa/polyglot/outer/nested-peak-mem-usage.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig"
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/nested-counters/polyglot-random-strings \
	--regex-file polyglot/nested-c-patterns-filtered.txt \
	--log-file outputs/csa/polyglot/inner/nested-peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" \
	--super-config-class "SparseCounterConfig"