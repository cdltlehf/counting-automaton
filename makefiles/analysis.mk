ANALYSIS_DIR := $(DATA_DIR)/analysis

STATIC_ANALYSIS_DIR := $(ANALYSIS_DIR)/static
DYNAMIC_ANALYSIS_DIR := $(ANALYSIS_DIR)/dynamic
COMPUTATION_INFO_DIR := $(DYNAMIC_ANALYSIS_DIR)/computation-info
COMPUTATION_INFO := $(foreach test_case,$(notdir $(TEST_CASES)),\
	$(foreach method,$(METHODS),$(COMPUTATION_INFO_DIR)/$(method)-$(basename $(test_case)).jsonl)\
)

.PHONY: static-analysis
static-analysis: $(FILTERED_PATTERNS)
	$(foreach filtered_pattern, $(FILTERED_PATTERNS), \
		$(PYTHON) $(PYFLAGS) scripts/analysis_pattern.py \
		--output-dir $(STATIC_ANALYSIS_DIR)/$(notdir $(basename $(filtered_pattern))) \
		< $(filtered_pattern); \
	)

define COMPUTATION_INFO_RULE
$(COMPUTATION_INFO_DIR)/$1-%.jsonl: $(TEST_CASES_DIR)/%.jsonl
	@mkdir -p $$(dir $$@)
	$(PYTHON) $(PYFLAGS) scripts/analysis/computation_info.py \
		--method $1 < $$< > $$@ 2> $$(basename $$@).log
endef

$(foreach method,$(METHODS),$(eval $(call COMPUTATION_INFO_RULE,$(method))))

.PHONY: computation-info
computation-info: $(COMPUTATION_INFO)
