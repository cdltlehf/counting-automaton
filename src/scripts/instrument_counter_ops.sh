mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
python -O -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir regex-input-gen/generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/snort3/inner/csa/op-name-to-counts-snort3.tsv \
	--merge-sizes-output outputs/snort3/inner/csa/merge-sizes-snort3.npy \
	--clone-sizes-output outputs/snort3/inner/csa/clone-sizes-snort3.npy \
	--expansion-type inner
python -O -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir regex-input-gen/generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/polyglot/inner/csa/op-name-to-counts-polyglot.tsv \
	--merge-sizes-output outputs/polyglot/inner/csa/merge-sizes-polyglot.npy \
	--clone-sizes-output outputs/polyglot/inner/csa/clone-sizes-polyglot.npy \
	--expansion-type inner
# TODO: run with attack strings as well?