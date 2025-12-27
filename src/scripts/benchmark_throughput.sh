mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/ce/polyglot/full
mkdir -p outputs/bva/polyglot/inner

POLYGLOT_RANDOM_STRINGS_DIR="generated/counters/polyglot-random-strings"
POLYGLOT_REGEX_FILE="polyglot/c-patterns-filtered.txt"


# Benchmark standard bitvector and counting-set approaches
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE\
	--timing-log-file outputs/ce/polyglot/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE\
	--timing-log-file outputs/csa/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR \
	--regex-file $POLYGLOT_REGEX_FILE\
	--timing-log-file outputs/bva/polyglot/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "bitvector" &

wait

# Do the same for SNORT3

mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/ce/snort3/full
mkdir -p outputs/bva/snort3/inner

SNORT3_RANDOM_STRINGS_DIR="generated/counters/snort3-random-strings"
SNORT3_REGEX_FILE="snort3/c-patterns-filtered.txt"

# Benchmark inner vs outer expansion for counting-set approach
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE\
	--timing-log-file outputs/csa/snort3/outer/random-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
# No need to repeat the inner expansion here since it was already done above

# Benchmark standard bitvector and counting-set approaches
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE\
	--timing-log-file outputs/ce/snort3/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE\
	--timing-log-file outputs/csa/snort3/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $SNORT3_RANDOM_STRINGS_DIR \
	--regex-file $SNORT3_REGEX_FILE\
	--timing-log-file outputs/bva/snort3/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "bitvector" &
wait