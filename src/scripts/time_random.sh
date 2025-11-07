mkdir -p ../outputs/csa/snort3/inner
mkdir -p ../outputs/csa/polyglot/inner
mkdir -p ../outputs/dfa/snort3/inner
mkdir -p ../outputs/dfa/polyglot/inner
# SNORT3
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/snort3/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"
# Outer
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/snort3/outer/random-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/snort3/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"


# Polyglot
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/polyglot/outer/random-throughputs-flush-on-full-cache.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"

# SNORT3
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/snort3/inner/random-throughputs-flush-on-full-cache.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../snort3/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/snort3/outer/random-throughputs-flush-on-full-cache.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"

# Polyglot
# Inner
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/polyglot/inner/random-throughputs-flush-on-full-cache.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
# Outer
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/polyglot/outer/random-throughputs-flush-on-full-cache.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full"
# Full
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file ../polyglot/c-patterns-filtered.txt \
	--timing-log-file ../outputs/csa/polyglot/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none"