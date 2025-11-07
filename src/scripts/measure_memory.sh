mkdir -p outputs/csa/inner/polyglot
mkdir -p outputs/csa/inner/snort3

# SNORT3
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--log-file outputs/csa/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

# Polyglot
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--log-file outputs/csa/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

mkdir -p outputs/dfa/snort3/inner
mkdir -p outputs/dfa/polyglot/inner

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--log-file outputs/dfa/snort3/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

# Polyglot
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--log-file outputs/dfa/polyglot/inner/peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30