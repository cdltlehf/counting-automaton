POLYGLOT_RANDOM_STRINGS_DIR="generated/ambiguous/polyglot-random-strings"
POLYGLOT_AMBIGUOUS="polyglot/ambiguous.txt"
POLYGLOT_UNAMBIGUOUS="polyglot/unambiguous.txt"

python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/inner/ambiguous/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" &

python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/inner/unambiguous/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" &

wait

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS\
	--memory-log-file outputs/csa/polyglot/inner/ambiguous/memory-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" &

python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS\
	--memory-log-file outputs/csa/polyglot/inner/unambiguous/memory-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--counter-type "counting-set" &

wait