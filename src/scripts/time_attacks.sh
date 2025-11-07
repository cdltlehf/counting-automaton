mkdir -p outputs/snort3
mkdir -p outputs/polyglot
# SNORT3
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/snort3-attack-strings \
	--regex-file ../../re-datasets/snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/snort3/inner-attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/snort3-attack-strings \
	--regex-file ../../re-datasets/snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/snort3/outer-attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/RegexStaticAnalysis/snort3 \
	--regex-file ../../re-datasets/snort3/c-patterns-filtered.txt \
	--timing-log-file outputs/snort3/outer-RegexStaticAnalysis-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8"

# Polyglot
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/polyglot-attack-strings \
	--regex-file ../../re-datasets/polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/polyglot/inner-attack-throughputs.tsv \
	--expansion-type inner \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/polyglot-attack-strings \
	--regex-file ../../re-datasets/polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/polyglot/outer-attack-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "latin1"
python -O -m cai4py.instrumentation.time_attacks \
	--attack-string-dir ../regex-input-gen/generated/counters/RegexStaticAnalysis/polyglot \
	--regex-file ../../re-datasets/polyglot/c-patterns-filtered.txt \
	--timing-log-file outputs/polyglot/outer-RegexStaticAnalysis-throughputs.tsv \
	--expansion-type outer \
	--input-encoding "utf-8"