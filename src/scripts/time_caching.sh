mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/csa/snort3/full
mkdir -p outputs/csa/polyglot/full

# Flush on full cache
# SNORT3
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
# Polyglot
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"

# LRU cache
# SNORT3
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "lru"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "lru"
# Polyglot
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "lru"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "lru"