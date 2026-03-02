mkdir -p outputs/csa/polyglot-bounds-le64/inner
mkdir -p outputs/csa/polyglot-bounds-gt64/inner
mkdir -p outputs/ce/polyglot-bounds-le64/full
mkdir -p outputs/ce/polyglot-bounds-gt64/full
mkdir -p outputs/bva/polyglot-bounds-le64/inner
mkdir -p outputs/bva/polyglot-bounds-gt64/inner
mkdir -p logs

POLYGLOT_RANDOM_STRINGS_DIR_LE64="generated/ambiguous/polyglot-random-strings-bounds-le64"
POLYGLOT_RANDOM_STRINGS_DIR_GT64="generated/ambiguous/polyglot-random-strings-bounds-gt64"
POLYGLOT_REGEX_FILE_LE64="polyglot/c-patterns-filtered-ambiguous-bounds-le64.txt"
POLYGLOT_REGEX_FILE_GT64="polyglot/c-patterns-filtered-ambiguous-bounds-gt64.txt"


# Counter expansion
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_LE64 \
	--regex-file $POLYGLOT_REGEX_FILE_LE64\
	--timing-log-file outputs/ce/polyglot-bounds-le64/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/throughput-polyglot-ce-full-le64.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_GT64 \
	--regex-file $POLYGLOT_REGEX_FILE_GT64\
	--timing-log-file outputs/ce/polyglot-bounds-gt64/full/random-throughputs.tsv \
	--expansion-type full \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/throughput-polyglot-ce-full-gt64.txt 2>&1 &

# Counting-set automaton
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_GT64 \
	--regex-file $POLYGLOT_REGEX_FILE_GT64\
	--timing-log-file outputs/csa/polyglot-bounds-gt64/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/throughput-polyglot-csa-inner-gt64.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_LE64 \
	--regex-file $POLYGLOT_REGEX_FILE_LE64\
	--timing-log-file outputs/csa/polyglot-bounds-le64/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/throughput-polyglot-csa-inner-le64.txt 2>&1 &

# Bitvector automaton
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_GT64 \
	--regex-file $POLYGLOT_REGEX_FILE_GT64\
	--timing-log-file outputs/bva/polyglot-bounds-gt64/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" > logs/throughput-polyglot-bva-inner-gt64.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_random \
	--random-string-dir $POLYGLOT_RANDOM_STRINGS_DIR_LE64 \
	--regex-file $POLYGLOT_REGEX_FILE_LE64\
	--timing-log-file outputs/bva/polyglot-bounds-le64/inner/random-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 5 \
	--cache-type "none" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" > logs/throughput-polyglot-bva-inner-le64.txt 2>&1 &

wait