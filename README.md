# EDLines Study Project

This project compares two **EDLines implementations**:
- Custom version in the root folder (`EDStudy`)
- Original EDLib as a **git submodule** in `ED_Lib/`

---

## Structure

```
.
├── ED_Lib/        # Original EDLines library
├── images/        # Test images
├── CMakeLists.txt # Build config
├── Makefile       # Init, build, run
├── test_ED.cpp    # Main test program
└── README.md
```

---

## Requirements

- **CMake ≥ 3.11**
- **g++** or **clang** with C++11
- **OpenCV**

### Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

---

## Setup

```bash
git clone https://github.com/<username>/EDLINES-IMPLEMENTATION.git
cd EDLINES-IMPLEMENTATION
make init
```

---

## Build

```bash
make build
```

- Builds custom project in `build/`
- Builds EDLib submodule in `ED_Lib/build/`

---

## Run

Run the original and new versions with:

```bash
make run IMAGE=<image_filename>
```

Results are saved in `results/`.

---

## Clean

```bash
make clean     # Remove builds and results
make rebuild   # Clean and build again
```
