# Compiler and flags
MPICC = mpicc
CFLAGS = -O3 -std=c99 -Wall -Wextra -fopenmp
LDFLAGS = -lm -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BIN_DIR)/cholesky_mpi

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) benchmarks

# Link object files
$(TARGET): $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(MPICC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean everything including benchmarks
distclean: clean
	rm -rf benchmarks
	@echo "Distribution clean complete"

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  clean      - Remove build files"
	@echo "  distclean  - Remove build files and benchmarks"
	@echo "  help       - Show this help message"

.PHONY: all directories clean distclean help
