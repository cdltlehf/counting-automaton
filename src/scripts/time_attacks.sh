mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/bva/snort3/inner
mkdir -p outputs/bva/polyglot/inner
mkdir -p logs

# SNORT3 - Counting-set automaton
python3 -OO -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/snort3/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/attacks-snort3-csa-inner.txt 2>&1 &
python3 -OO -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/ambiguous/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/snort3/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/attacks-snort3-csa-RegexStaticAnalysis-inner.txt 2>&1 &

# Polyglot - Counting-set automaton
python3 -OO -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/polyglot/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/attacks-polyglot-csa-inner.txt 2>&1 &
python3 -OO -m cai4py.instrumentation.time_attacks \
	--super-config-class "SparseCounterConfig" \
	--attack-string-dir generated/ambiguous/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/csa/polyglot/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--counter-type "counting-set" \
	--cache-type "none" > logs/attacks-polyglot-csa-RegexStaticAnalysis-inner.txt 2>&1 &


# SNORT3 - Bitvector automaton
python3 -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/bva/snort3/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" \
	--cache-type "none" > logs/attacks-snort3-bva-inner.txt 2>&1 &
python3 -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/bva/snort3/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" \
	--cache-type "none" > logs/attacks-snort3-bva-RegexStaticAnalysis-inner.txt 2>&1 &

# Polyglot - Bitvector automaton
python3 -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/bva/polyglot/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" \
	--cache-type "none" > logs/attacks-polyglot-bva-inner.txt 2>&1 &
python3 -OO -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/ambiguous/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--timing-log-file outputs/bva/polyglot/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--super-config-class "SuperConfig" \
	--counter-type "bitvector" \
	--cache-type "none" > logs/attacks-polyglot-bva-RegexStaticAnalysis-inner.txt 2>&1 &

wait