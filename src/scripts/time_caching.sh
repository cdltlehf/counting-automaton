# Create necessary output directories
mkdir -p outputs/csa/snort3/full
mkdir -p outputs/csa/polyglot/full
mkdir -p logs

# Flush on full cache - Full expansion only
# SNORT3
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/snort3/full/attack-throughputs-flush_on_full.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "flush_on_full" > logs/caching-snort3-full-flush_on_full.txt 2>&1 &
# Polyglot
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/polyglot/full/attack-throughputs-flush_on_full.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "flush_on_full" > logs/caching-polyglot-full-flush_on_full.txt 2>&1 &

# LRU cache - Full expansion only
# SNORT3
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/snort3/full/attack-throughputs-lru.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "lru" > logs/caching-snort3-full-lru.txt 2>&1 &
# Polyglot
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/polyglot/full/attack-throughputs-lru.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "lru" > logs/caching-polyglot-full-lru.txt 2>&1 &

# No caching - Full expansion only
# SNORT3
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/snort3/full/attack-throughputs-none.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/caching-snort3-full-none.txt 2>&1 &
# Polyglot
python -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/polyglot/full/attack-throughputs-none.tsv \
	--expansion-type full \
	--input-encoding "latin1" \
	--super-config-class "SparseCounterConfig" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/caching-polyglot-full-none.txt 2>&1 &
wait