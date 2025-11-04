mkdir -p outputs/csa
python -O -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir ../regex-input-gen/generated/counters/snort3-random-strings \
	--regex-file ../../re-datasets/snort3/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/csa/op-name-to-counts-snort3.tsv \
	--merge-sizes-output outputs/csa/merge-sizes-snort3.npy \
	--clone-sizes-output outputs/csa/clone-sizes-snort3.npy \
	--expansion-type inner
python -O -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir ../regex-input-gen/generated/counters/polyglot-random-strings \
	--regex-file ../../re-datasets/polyglot/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/csa/op-name-to-counts-polyglot.tsv \
	--merge-sizes-output outputs/csa/merge-sizes-polyglot.npy \
	--clone-sizes-output outputs/csa/clone-sizes-polyglot.npy \
	--expansion-type inner
# TODO: run with attack strings as well?