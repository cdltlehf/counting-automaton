POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR="generated/unambiguous/polyglot-attack-strings"
POLYGLOT_UNAMBIGUOUS_RANDOM_STRING_DIR="generated/unambiguous/polyglot-random-strings"
POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR_100="generated/unambiguous/polyglot-attack-strings-100"
POLYGLOT_UNAMBIGUOUS="polyglot/c-patterns-filtered-unambiguous-sampled.txt"
POLYGLOT_UNAMBIGUOUS_100="polyglot/c-patterns-filtered-unambiguous-100-sampled.txt"

POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR="generated/ambiguous/polyglot-attack-strings"
POLYGLOT_AMBIGUOUS_RANDOM_STRING_DIR="generated/ambiguous/polyglot-random-strings"
POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR_100="generated/ambiguous/polyglot-attack-strings-100"
POLYGLOT_AMBIGUOUS="polyglot/c-patterns-filtered-ambiguous.txt"
POLYGLOT_AMBIGUOUS_100="polyglot/c-patterns-filtered-ambiguous-100-sampled.txt"
mkdir -p outputs/csa/polyglot/inner/ambiguous
mkdir -p outputs/csa/polyglot/inner/unambiguous

python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/ambiguous/attack-throughputs-cached.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "flush_on_full" \
	--sample-interval 2 \
	--cache-history-log-file outputs/csa/polyglot/inner/ambiguous/cache-timeline-attack.tsv \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set"
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/unambiguous/attack-throughputs-cached.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "flush_on_full" \
	--cache-history-log-file outputs/csa/polyglot/inner/unambiguous/cache-timeline-attack.tsv \
	--sample-interval 2 \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set"


# python -OO -m cai4py.instrumentation.time_random \
# 	--random-string-dir $POLYGLOT_AMBIGUOUS_RANDOM_STRING_DIR \
# 	--regex-file $POLYGLOT_AMBIGUOUS \
# 	--timing-log-file outputs/csa/polyglot/inner/ambiguous/random-throughputs-cached.tsv \
# 	--expansion-type inner \
# 	--input-encoding "utf-8" \
# 	--cache-type "flush_on_full" \
# 	--sample-interval 2 \
# 	--cache-history-log-file outputs/csa/polyglot/inner/ambiguous/cache-timeline-random.tsv \
# 	--super-config-class "SparseCounterConfig" \
# 	--num-strings-per-regex 1 \
# 	--counter-type "counting-set"
# python -OO -m cai4py.instrumentation.time_random \
# 	--random-string-dir $POLYGLOT_UNAMBIGUOUS_RANDOM_STRING_DIR \
# 	--regex-file $POLYGLOT_UNAMBIGUOUS \
# 	--timing-log-file outputs/csa/polyglot/inner/unambiguous/random-throughputs-cached.tsv \
# 	--expansion-type inner \
# 	--input-encoding "utf-8" \
# 	--cache-type "flush_on_full" \
# 	--cache-history-log-file outputs/csa/polyglot/inner/unambiguous/cache-timeline-random.tsv \
# 	--sample-interval 2 \
# 	--super-config-class "SparseCounterConfig" \
# 	--num-strings-per-regex 1 \
# 	--counter-type "counting-set"