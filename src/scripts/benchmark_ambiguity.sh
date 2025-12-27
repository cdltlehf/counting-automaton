POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR="generated/unambiguous/polyglot-attack-strings"
POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR_100="generated/unambiguous/polyglot-attack-strings-100"
POLYGLOT_UNAMBIGUOUS="polyglot/c-patterns-filtered-unambiguous-sampled.txt"
POLYGLOT_UNAMBIGUOUS_100="polyglot/c-patterns-filtered-unambiguous-100-sampled.txt"

POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR="generated/ambiguous/polyglot-attack-strings"
POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR_100="generated/ambiguous/polyglot-attack-strings-100"
POLYGLOT_AMBIGUOUS="polyglot/c-patterns-filtered-ambiguous.txt"
POLYGLOT_AMBIGUOUS_100="polyglot/c-patterns-filtered-ambiguous-100-sampled.txt"
mkdir -p outputs/csa/polyglot/inner/ambiguous
mkdir -p outputs/csa/polyglot/inner/unambiguous

python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/ambiguous/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/log1.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/unambiguous/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/log2.txt 2>&1 &

wait

python -OO -m cai4py.instrumentation.measure_worst_memory \
	--attack-string-dir $POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR_100 \
	--regex-file $POLYGLOT_AMBIGUOUS_100 \
	--log-file outputs/csa/polyglot/inner/ambiguous/memory-usage.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/log5.txt 2>&1 &
python -OO -m cai4py.instrumentation.measure_worst_memory \
	--attack-string-dir $POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR_100 \
	--regex-file $POLYGLOT_UNAMBIGUOUS_100 \
	--log-file outputs/csa/polyglot/inner/unambiguous/memory-usage.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" > logs/log6.txt 2>&1 &

wait

python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_AMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_AMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/ambiguous/attack-throughputs-cached.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "flush_on_full" \
	--super-config-class "SparseCounterConfig" \
	--sample-interval 2 \
	--cache-history-log-file outputs/csa/polyglot/inner/ambiguous/cache-timeline.tsv \
	--counter-type "counting-set" > logs/log3.txt 2>&1 &
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir $POLYGLOT_UNAMBIGUOUS_ATTACK_STRING_DIR \
	--regex-file $POLYGLOT_UNAMBIGUOUS \
	--timing-log-file outputs/csa/polyglot/inner/unambiguous/attack-throughputs-cached.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "flush_on_full" \
	--super-config-class "SparseCounterConfig" \
	--cache-history-log-file outputs/csa/polyglot/inner/unambiguous/cache-timeline.tsv \
	--sample-interval 2 \
	--counter-type "counting-set" > logs/log4.txt 2>&1 &

wait
echo "✅ Benchmark complete"