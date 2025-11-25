POLYGLOT_RANDOM_STRINGS_DIR="generated/ambiguous/polyglot-random-strings"
POLYGLOT_AMBIGUOUS="polyglot/ambiguous.txt"
POLYGLOT_UNAMBIGUOUS="polyglot/unambiguous.txt"

python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/full/ambiguous/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "none" &
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/full/unambiguous/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "none" &
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS\
	--log-file outputs/csa/polyglot/full/ambiguous/memory-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "none" &
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS\
	--log-file outputs/csa/polyglot/full/unambiguous/memory-usage.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "none" &
wait
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/full/ambiguous/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full" \
	--super-config-class "SuperConfig" \
	--sample-interval 2 \
	--counter-type "none" &
python -O -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS\
	--timing-log-file outputs/csa/polyglot/full/unambiguous/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "flush_on_full" \
	--super-config-class "SuperConfig" \
	--sample-interval 2 \
	--counter-type "none" &
