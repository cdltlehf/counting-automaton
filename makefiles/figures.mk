FIGURES_DIR := $(DATA_DIR)/figures
COUNTER_RANGE_DIR := $(FIGURES_DIR)/counter-range
COUNTER_RANGE := $(COUNTER_RANGE_DIR)/$(PATTERN_BASENAMES:.txt=.pdf)
COMPUTATION_COMPARISON_DIR := $(FIGURES_DIR)/computations-comparison

COMPUTATION_COMPARISON := \
$(foreach pattern_basename,$(PATTERN_BASENAMES:.txt=),\
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-bounded_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-sparse_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_bounded_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_sparse_counter_config.pdf \
	\
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-sparse_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-determinized_bounded_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-determinized_sparse_counter_config.pdf \
	\
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-sparse_counter_config-determinized_bounded_counter_config.pdf \
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-sparse_counter_config-determinized_sparse_counter_config.pdf \
	\
	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-determinized_bounded_counter_config-determinized_sparse_counter_config.pdf \
)

# COMPUTATION_COMPARISON := \
# $(foreach pattern_basename,$(PATTERN_BASENAMES:.txt=),\
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-bounded_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_bounded_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_sparse_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-sparse_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-determinized_bounded_counter_config.pdf \
# )

FORCE:

define COUNTER_RANGE_RULE
$(foreach ext,pdf png pgf,\
$(COUNTER_RANGE_DIR)/$1.$(ext)): $(TEST_CASES_DIR)/$1.txt FORCE
	@mkdir -p $$(dir $$@)
	- $(PYTHON) $(PYFLAGS) scripts/figures/plot_counter_range.py $$< $$@

endef

define COMPUTATION_COMPARISON_RULE
$(foreach ext,pdf png pgf,\
$(COMPUTATION_COMPARISON_DIR)/$1-$2-$3.$(ext)): \
		$(COMPUTATION_INFO_DIR)/$1-$2.jsonl \
		$(COMPUTATION_INFO_DIR)/$1-$3.jsonl \
		FORCE
	@mkdir -p $$(dir $$@)
	- $(PYTHON) $(PYFLAGS) scripts/figures/plot_computation_comparison.py \
		--x-label $2 \
		--y-label $3 \
		$(COMPUTATION_INFO_DIR)/$1-$2.jsonl \
		$(COMPUTATION_INFO_DIR)/$1-$3.jsonl \
		$$@ > $$(basename $$@).txt 2> $$(basename $$@)_log.txt

endef

$(eval $(foreach name,$(PATTERN_BASENAMES:.txt=),$(call COUNTER_RANGE_RULE,$(name))))
$(eval $(foreach name,$(PATTERN_BASENAMES:.txt=),\
	$(foreach method_x,$(METHODS),\
		$(foreach method_y,$(METHODS),\
			$(call COMPUTATION_COMPARISON_RULE,$(name),$(method_x),$(method_y))\
		)\
	)\
))

.PHONY: counter-range
counter-range: $(COUNTER_RANGE)

.PHONY: computation-comparison
computation-comparison: $(COMPUTATION_COMPARISON)
computation-comparison.zip: $(COMPUTATION_COMPARISON_DIR).zip

$(COMPUTATION_COMPARISON_DIR).zip: FORCE
	@rm -f $@
	@mkdir -p $(dir $@)
	zip -rj $@ $(dir $(COMPUTATION_COMPARISON))
