# =========================
# Makefile for EDLines Study Project
# =========================

# -------------------------
# Variables
# -------------------------
BUILD_DIR := build
EDLIB_DIR := ED_Lib
IMAGES_DIR := images
RESULTS_DIR := results
MAIN_EXE := $(BUILD_DIR)/test_ED
SUB_EXE := $(EDLIB_DIR)/build/test_ED

# -------------------------
# Default target
# -------------------------
.PHONY: all
all: init build

# -------------------------
# 🔹 Initialize repository
# -------------------------
.PHONY: init
init:
	@echo "📦 Initializing submodules..."
	git submodule update --init --recursive

# -------------------------
# 🔹 Build both projects
# -------------------------
.PHONY: build
build: build-main build-edlib

build-main:
	@echo "🔨 Building main project..."
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) || { echo "❌ CMake configuration failed for main project"; exit 1; }
	@cmake --build $(BUILD_DIR) -j$$(nproc) || { echo "❌ Build failed for main project"; exit 1; }

build-edlib:
	@echo "🔨 Building EDLib submodule..."
	@mkdir -p $(EDLIB_DIR)/build
	@cmake -S $(EDLIB_DIR) -B $(EDLIB_DIR)/build || { echo "❌ CMake configuration failed for EDLib"; exit 1; }
	@cmake --build $(EDLIB_DIR)/build -j$$(nproc) || { echo "❌ Build failed for EDLib"; exit 1; }

# -------------------------
# 🔹 Run both tests
# -------------------------
.PHONY: run
run: run-main run-edlib

run-main:
	@echo "🚀 Running test_ED from main project..."
	@mkdir -p $(RESULTS_DIR)
	@if [ -z "$(IMAGE)" ]; then \
		echo "❗ No image specified. Usage: make run-main IMAGE=<filename> (must exist in $(IMAGES_DIR))"; \
		echo "Available images:"; ls -1 $(IMAGES_DIR) || true; \
	elif [ ! -f "$(MAIN_EXE)" ]; then \
		echo "❌ Executable $(MAIN_EXE) not found. Run 'make build' first."; \
	elif [ ! -f "$(IMAGES_DIR)/$(IMAGE)" ]; then \
		echo "❌ Image '$(IMAGE)' not found in $(IMAGES_DIR)."; \
	else \
		echo "▶️ Running with $(IMAGES_DIR)/$(IMAGE)"; \
		cd $(RESULTS_DIR) && ../$(MAIN_EXE) ../$(IMAGES_DIR)/$(IMAGE); \
	fi

run-edlib:
	@echo "🚀 Running test_ED from ED_Lib submodule..."
	@mkdir -p $(RESULTS_DIR)
	@if [ -z "$(IMAGE)" ]; then \
		echo "❗ No image specified. Usage: make run-edlib IMAGE=<filename> (must exist in $(IMAGES_DIR))"; \
		echo "Available images:"; ls -1 $(IMAGES_DIR) || true; \
	elif [ ! -f "$(SUB_EXE)" ]; then \
		echo "❌ Executable $(SUB_EXE) not found. Run 'make build' first."; \
	elif [ ! -f "$(IMAGES_DIR)/$(IMAGE)" ]; then \
		echo "❌ Image '$(IMAGE)' not found in $(IMAGES_DIR)."; \
	else \
		echo "▶️ Running with $(IMAGES_DIR)/$(IMAGE)"; \
		cd $(RESULTS_DIR) && ../$(SUB_EXE) ../$(IMAGES_DIR)/$(IMAGE); \
	fi


# -------------------------
# 🔹 Clean builds
# -------------------------
.PHONY: clean
clean:
	@echo "🧹 Cleaning all build files..."
	rm -rf $(BUILD_DIR)
	rm -rf $(EDLIB_DIR)/build
	rm -rf $(RESULTS_DIR)

# -------------------------
# 🔹 Full rebuild
# -------------------------
.PHONY: rebuild
rebuild: clean all
