mkdir -p outputs/csa/snort3/inner
mkdir -p outputs/csa/polyglot/inner
mkdir -p logs

python -OO -m cai4py.instrumentation.instrument_counter_ops_attack \
	--attack-string-dir generated/ambiguous/snort3-attack-strings \
	--regex-file snort3/c-patterns-filtered-ambiguous.txt \
	--op-counts-output outputs/csa/snort3/inner/op-name-to-counts.tsv \
	--merge-sizes-output outputs/csa/snort3/inner/merge-sizes.npy \
	--clone-sizes-output outputs/csa/snort3/inner/clone-sizes.npy \
	--input-encoding "latin1" \
	--expansion-type inner > logs/instrument-snort3-csa-inner.txt 2>&1 &
python -OO -m cai4py.instrumentation.instrument_counter_ops_attack \
	--attack-string-dir generated/ambiguous/polyglot-attack-strings \
	--regex-file polyglot/c-patterns-filtered-ambiguous.txt \
	--op-counts-output outputs/csa/polyglot/inner/op-name-to-counts.tsv \
	--merge-sizes-output outputs/csa/polyglot/inner/merge-sizes.npy \
	--clone-sizes-output outputs/csa/polyglot/inner/clone-sizes.npy \
	--input-encoding "latin1" \
	--expansion-type inner > logs/instrument-polyglot-csa-inner.txt 2>&1 &
wait
# TODO: run with attack strings as well?