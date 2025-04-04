ANALYSIS_DIR := $(DATA_DIR)/analysis

STATIC_ANALYSIS_DIR := $(ANALYSIS_DIR)/static
DYNAMIC_ANALYSIS_DIR := $(ANALYSIS_DIR)/dynamic
COMPUTATION_INFO_DIR := $(DYNAMIC_ANALYSIS_DIR)/computation-info
COMPUTATION_INFO := \
$(foreach pattern_basename,$(PATTERN_BASENAMES:.txt=),\
	$(foreach method,$(METHODS),\
		$(COMPUTATION_INFO_DIR)/$(pattern_basename)-$(method).jsonl\
	)\
)

.PHONY: static-analysis
static-analysis: $(FILTERED_PATTERNS)
	$(foreach filtered_pattern, $(FILTERED_PATTERNS), \
		$(PYTHON) $(PYFLAGS) scripts/analysis_pattern.py \
		--output-dir $(STATIC_ANALYSIS_DIR)/$(notdir $(basename $(filtered_pattern))) \
		< $(filtered_pattern); \
	)

define COMPUTATION_INFO_RULE
$(COMPUTATION_INFO_DIR)/$1-$2.jsonl: $(TEST_CASES_DIR)/$1.txt
	@mkdir -p $$(dir $$@)
	- $(PYTHON) $(PYFLAGS) scripts/analysis/computation_info.py \
		--method $2 \
		< $$< \
		> $$@ 2> /dev/null

endef

$(eval $(foreach pattern_basename,$(PATTERN_BASENAMES:.txt=),\
	$(foreach method,$(METHODS),\
		$(call COMPUTATION_INFO_RULE,$(pattern_basename),$(method))\
	)\
))

.PHONY: computation-info
computation-info: $(COMPUTATION_INFO)
