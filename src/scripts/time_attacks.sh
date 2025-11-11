mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
mkdir -p outputs/csa/snort3/outer
mkdir -p outputs/csa/polyglot/outer

# SNORT3
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner-attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/outer-attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/outer-RegexStaticAnalysis-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/RegexStaticAnalysis/snort3 \
	--regex-file snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/snort3/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--cache-type "none"
sleep 5

# Polyglot
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/outer/attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/outer/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--cache-type "none"
sleep 5
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir generated/counters/RegexStaticAnalysis/polyglot \
	--regex-file polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/csa/polyglot/inner/RegexStaticAnalysis-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--cache-type "none"