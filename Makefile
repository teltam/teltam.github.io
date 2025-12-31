# Variables
MARKDOWN_DIR = md
HTML_DIR = posts
CSS_FILE = /css/water.min.css
PANDOC = pandoc
PANDOC_OPTIONS = -H cf --standalone --mathjax --highlight-style=tango
PORT = 8000

# Find all Markdown files
MARKDOWN_FILES = $(wildcard $(MARKDOWN_DIR)/*.md)

# Generate HTML filenames
HTML_FILES = $(patsubst $(MARKDOWN_DIR)/%.md,$(HTML_DIR)/%.html,$(MARKDOWN_FILES))

# Default target
all: generate-index $(HTML_FILES) index.html

# Generate index.md from md/ files
generate-index:
	python3 scripts/generate_index.py

# Rule to convert Markdown to HTML
$(HTML_DIR)/%.html: $(MARKDOWN_DIR)/%.md
	@mkdir -p $(HTML_DIR)
	$(PANDOC) $(PANDOC_OPTIONS) \
		--css=$(CSS_FILE) \
		--resource-path=$(MARKDOWN_DIR):media \
		-f markdown -t html \
		-o $@ $<

# Special rule for index.md
index.html: index.md
	$(PANDOC) $(PANDOC_OPTIONS) \
		--css=$(CSS_FILE) \
		--resource-path=.:media \
		-f markdown -t html \
		-o $@ $<

# Clean target
clean:
	rm -rf $(HTML_DIR)
	rm -f index.html

# Serve target
serve: all
	@echo "Starting HTTP server on port $(PORT)..."
	@python3 -m http.server $(PORT)

# Phony targets
.PHONY: all clean serve generate-index
