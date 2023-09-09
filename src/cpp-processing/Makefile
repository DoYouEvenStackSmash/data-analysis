# Makefile for test_tree.cpp

# Compiler and flags
CC = g++
CFLAGS = -std=c++11 -Wall

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# Executable name
EXECUTABLE = $(BIN_DIR)/test_tree

# Compilation rules
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean