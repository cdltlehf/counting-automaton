mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/csa/snort3/inner

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--log-file outputs/csa/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered-small.txt \
	--log-file outputs/csa/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

# Counter expansion
mkdir -p outputs/csa/snort3/full
mkdir -p outputs/csa/polyglot/full

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--log-file outputs/csa/snort3/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered-small.txt \
	--log-file outputs/csa/polyglot/full/peak-mem-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30