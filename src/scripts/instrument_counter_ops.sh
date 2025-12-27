mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
python -OO -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir generated/counters/snort3-random-strings \
	--regex-file snort3/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/csa/snort3/inner/op-name-to-counts.tsv \
	--merge-sizes-output outputs/csa/snort3/inner/merge-sizes.npy \
	--clone-sizes-output outputs/csa/snort3/inner/clone-sizes.npy \
	--expansion-type inner &
python -OO -m cai4py.instrumentation.instrument_counter_ops \
	--random-string-dir generated/counters/polyglot-random-strings \
	--regex-file polyglot/c-patterns-filtered.txt \
	--num-strings-per-regex 30 \
	--op-counts-output outputs/csa/polyglot/inner/op-name-to-counts.tsv \
	--merge-sizes-output outputs/csa/polyglot/inner/merge-sizes.npy \
	--clone-sizes-output outputs/csa/polyglot/inner/clone-sizes.npy \
	--expansion-type inner &
wait
# TODO: run with attack strings as well?