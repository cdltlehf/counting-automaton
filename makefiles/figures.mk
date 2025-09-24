FIGURES_DIR := $(DATA_DIR)/figures
COUNTER_RANGE_DIR := $(FIGURES_DIR)/counter-range
COUNTER_RANGE := $(COUNTER_RANGE_DIR)/$(PATTERN_BASENAMES:.txt=.pdf)

COMPUTATION_COMPARISON_DIR := $(FIGURES_DIR)/computations-comparison
MATCHING_TIME_COMPARISON_DIR := $(FIGURES_DIR)/matching-time-comparison
CASE_STUDY_DIR := $(FIGURES_DIR)/case-study

define triangular-loop
$(foreach j,$(wordlist 2,$(words $(1)),$(1)),$(firstword $(1))-$(j)) \
$(if $(wordlist 2,$(words $(1)),$(1)),$(call triangular-loop,$(wordlist 2,$(words $(1)),$(1))))
endef

PAIRS := $(call triangular-loop,$(METHODS))
COMPUTATION_COMPARISON := \
$(foreach method_xy,$(PAIRS),\
	$(foreach test_case,$(basename $(notdir $(TEST_CASES))),\
		$(COMPUTATION_COMPARISON_DIR)/$(method_xy)-$(test_case).pdf\
	)\
)

MATCHING_TIME_COMPARISON := \
$(foreach method_xy,$(PAIRS),\
	$(foreach test_case,$(basename $(notdir $(TEST_CASES))),\
		$(MATCHING_TIME_COMPARISON_DIR)/$(method_xy)-$(test_case).pdf\
	)\
)

# METHODS_X := bounded_super_config
# METHODS_Y := $(METHODS)
# COMPUTATION_COMPARISON := \
# $(foreach method_x,$(METHODS_X),\
# 	$(foreach method_y,$(METHODS_Y),\
# 		$(foreach test_case,$(basename $(notdir $(TEST_CASES))),\
# 			$(COMPUTATION_COMPARISON_DIR)/$(method_x)-$(method_y)-$(test_case).pdf\
# 		)\
# 	)\
# )

# COMPUTATION_COMPARISON := \
# $(foreach pattern_basename,$(PATTERN_BASENAMES:.txt=),\
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-bounded_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_bounded_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_super_config-determinized_sparse_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-sparse_counter_config.pdf \
# 	$(COMPUTATION_COMPARISON_DIR)/$(pattern_basename)-bounded_counter_config-determinized_bounded_counter_config.pdf \
# )

define COUNTER_RANGE_RULE
$(foreach ext,pdf png pgf,\
$(COUNTER_RANGE_DIR)/$1.$(ext)): $(TEST_CASES_DIR)/$1.txt
	@mkdir -p $$(dir $$@)
	- $(PYTHON) $(PYFLAGS) scripts/figures/plot_counter_range.py $$< $$@

endef

define COMPUTATION_COMPARISON_RULE
$(COMPUTATION_COMPARISON_DIR)/$1-$2-%.pdf: \
		$(COMPUTATION_INFO_DIR)/$1-%.jsonl \
		$(COMPUTATION_INFO_DIR)/$2-%.jsonl
	@mkdir -p $$(dir $$@)
	$(PYTHON) $(PYFLAGS) scripts/figures/plot_computation_comparison.py \
		--x-label $1 --y-label $2 \
		computation-steps \
		$(COMPUTATION_INFO_DIR)/$1-$$*.jsonl \
		$(COMPUTATION_INFO_DIR)/$2-$$*.jsonl \
		$$@ > $$(basename $$@).txt
endef

$(foreach method_y,$(METHODS),\
	$(foreach method_x,$(METHODS),\
		$(eval $(call COMPUTATION_COMPARISON_RULE,$(method_x),$(method_y)))\
	)\
)

define MATCHING_TIME_COMPARISON_RULE
$(MATCHING_TIME_COMPARISON_DIR)/$1-$2-%.pdf: \
		$(COMPUTATION_INFO_DIR)/$1-%.jsonl \
		$(COMPUTATION_INFO_DIR)/$2-%.jsonl
	@mkdir -p $$(dir $$@)
	$(PYTHON) $(PYFLAGS) scripts/figures/plot_computation_comparison.py \
		--x-label $1 --y-label $2 \
		matching-time \
		$(COMPUTATION_INFO_DIR)/$1-$$*.jsonl \
		$(COMPUTATION_INFO_DIR)/$2-$$*.jsonl \
		$$@ > $$(basename $$@).txt
endef

$(foreach method_y,$(METHODS),\
	$(foreach method_x,$(METHODS),\
		$(eval $(call MATCHING_TIME_COMPARISON_RULE,$(method_x),$(method_y)))\
	)\
)

$(eval $(foreach name,$(PATTERN_BASENAMES:.txt=),$(call COUNTER_RANGE_RULE,$(name))))

case-study:
	@mkdir -p $(CASE_STUDY_DIR)
	$(PYTHON) $(PYFLAGS) scripts/figures/plot_case_study.py \
		$(CASE_STUDY_DIR)

.PHONY: counter-range
counter-range: $(COUNTER_RANGE)

.PHONY: computation-comparison
computation-comparison:
	rm -rf $(COMPUTATION_COMPARISON_DIR)
	$(MAKE) $(COMPUTATION_COMPARISON)

.PHONY: matching-time-comparison
matching-time-comparison:
	rm -rf $(MATCHING_TIME_COMPARISON_DIR)
	$(MAKE) $(MATCHING_TIME_COMPARISON)


computation-comparison.zip: $(COMPUTATION_COMPARISON_DIR).zip
	mv $< $@

matching-time-comparison.zip: $(MATCHING_TIME_COMPARISON_DIR).zip
	mv $< $@

.PHONY: $(COMPUTATION_COMPARISON_DIR).zip
$(COMPUTATION_COMPARISON_DIR).zip:
	$(MAKE) computation-comparison
	@rm -f $@
	@mkdir -p $(dir $@)
	zip -rj $@ $(dir $(COMPUTATION_COMPARISON))

.PHONY: $(MATCHING_TIME_COMPARISON_DIR).zip
$(MATCHING_TIME_COMPARISON_DIR).zip:
	$(MAKE) matching-time-comparison
	@rm -f $@
	@mkdir -p $(dir $@)
	zip -rj $@ $(dir $(MATCHING_TIME_COMPARISON_DIR))
