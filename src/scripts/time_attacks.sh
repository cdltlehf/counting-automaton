# SNORT3
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/snort3-attack-strings \
	--regex-file ../../datasets/snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/inner-snort3-attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/snort3-attack-strings \
	--regex-file ../../datasets/snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/outer-snort3-attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1"

# Polyglot
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/polyglot-attack-strings \
	--regex-file ../../datasets/polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/inner-polyglot-attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/polyglot-attack-strings \
	--regex-file ../../datasets/polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/outer-polyglot-attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1"
