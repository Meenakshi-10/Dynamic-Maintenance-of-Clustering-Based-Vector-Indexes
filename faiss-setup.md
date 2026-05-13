# Setup (macOS)

These are the commands I ran to set up the FAISS fork for this research project on macOS (Apple Silicon).

## 1. Fork and clone

Forked `facebookresearch/faiss` on GitHub, then:

```bash
git clone https://github.com/<your-username>/faiss.git
cd faiss
git remote add upstream https://github.com/facebookresearch/faiss.git
git fetch upstream --tags
```

## 2. Pin to a stable release and create research branch

```bash
git tag -l "v*" | tail -5
git checkout -b ivf-mutable v1.12.0
```

## 3. Install dependencies

```bash
brew install libomp cmake swig openblas
```

## 4. Configure the build

Apple's clang doesn't ship with OpenMP, so libomp paths need to be passed explicitly:

```bash
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib" \
  .
```

## 5. Build the core library

```bash
cd build
make -j$(sysctl -n hw.ncpu) faiss
```

Output should produce `build/faiss/libfaiss.a` (or `libfaiss.dylib`).