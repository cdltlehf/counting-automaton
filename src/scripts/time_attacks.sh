mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/bva/snort3/inner
mkdir -p outputs/bva/polyglot/inner

# SNORT3 - Counting-set automaton
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "counting-set" \
	--cache-type "none" &
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "counting-set" \
	--cache-type "none" &

# Polyglot - Counting-set automaton
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "counting-set" \
	--cache-type "none" &
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "counting-set" \
	--cache-type "none" &


# SNORT3 - Bitvector automaton
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/bva/snort3/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "bitvector" \
	--cache-type "none" &
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/bva/snort3/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "bitvector" \
	--cache-type "none" &

# Polyglot - Bitvector automaton
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/bva/polyglot/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "bitvector" \
	--cache-type "none" &
python3 -O -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/counters/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/bva/polyglot/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "bitvector" \
	--cache-type "none" &

wait