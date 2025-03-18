ANALYSIS_DIR := $(DATA_DIR)/analysis

STATIC_ANALYSIS_DIR := $(ANALYSIS_DIR)/static
# STATIC_ANALYSIS := $(subst $(FILTERED_PATTERNS_DIR),$(STATIC_ANALYSIS_DIR), $(FILTERED_PATTERNS:.txt=.json))

# $(STATIC_ANALYSIS_DIR)/%.json: $(FILTERED_PATTERNS_DIR)/%.txt
# 	@mkdir -p $(dir $@)
# 	# $(PYTHON) scripts/analysis_pattern.py < $< > $@ || rm $@
# 	$(PYTHON) scripts/analysis_pattern.py < $<

.PHONY: static-analysis
static-analysis: $(FILTERED_PATTERNS)
	$(foreach filtered_pattern, $(FILTERED_PATTERNS), \
		$(PYTHON) scripts/analysis_pattern.py \
		--output-dir $(STATIC_ANALYSIS_DIR)/$(notdir $(basename $(filtered_pattern))) \
		< $(filtered_pattern); \
	)
