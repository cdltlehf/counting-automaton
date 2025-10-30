# SNORT3
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir ../regex-input-gen/generated/counters/snort3-random-strings \
	--regex-file ../../datasets/snort3/c-patterns-filtered.txt \
	--log-file outputs/inner-snort3-peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir ../regex-input-gen/generated/counters/snort3-random-strings \
	--regex-file ../../datasets/snort3/c-patterns-filtered.txt \
	--log-file outputs/outer-snort3-peak-mem-usage.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30

# Polyglot
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir ../regex-input-gen/generated/counters/polyglot-random-strings \
	--regex-file ../../datasets/polyglot/c-patterns-filtered.txt \
	--log-file outputs/inner-polyglot-peak-mem-usage.tsv \
	--expansion-type inner \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30
python -O -m cai4py.instrumentation.measure_memory \
	--random-string-dir ../regex-input-gen/generated/counters/polyglot-random-strings \
	--regex-file ../../datasets/polyglot/c-patterns-filtered.txt \
	--log-file outputs/outer-polyglot-peak-mem-usage.tsv \
	--expansion-type outer \
	--input-encoding "utf-8" \
	--num-strings-per-regex 30