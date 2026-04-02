import os
import tarfile
import urllib.request as request

import numpy as np
import torch
import quake

# ── Config ────────────────────────────────────────────────────────────────────

SIFT_URL     = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_ARCHIVE = "sift.tar.gz"
SIFT_DIR     = "sift"

NLIST  = 1024
METRIC = "l2"

# ── Download & extract ────────────────────────────────────────────────────────

def download_sift():
    if not os.path.exists(SIFT_ARCHIVE):
        print("Downloading SIFT1M...")
        request.urlretrieve(SIFT_URL, SIFT_ARCHIVE)

    if not os.path.exists(f"{SIFT_DIR}/sift_base.fvecs"):
        print("Extracting archive...")
        with tarfile.open(SIFT_ARCHIVE, "r:gz") as tar:
            tar.extractall()

# ── Readers ───────────────────────────────────────────────────────────────────

def read_fvecs(filename):
    with open(filename, "rb") as f:
        d    = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(-1, d + 1)[:, 1:]

def read_ivecs(filename):
    with open(filename, "rb") as f:
        d    = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
    return data.reshape(-1, d + 1)[:, 1:]

# ── Load data ─────────────────────────────────────────────────────────────────

def load_sift1m():
    download_sift()
    xb     = read_fvecs(f"{SIFT_DIR}/sift_base.fvecs")
    xq     = read_fvecs(f"{SIFT_DIR}/sift_query.fvecs")
    I_true = read_ivecs(f"{SIFT_DIR}/sift_groundtruth.ivecs")
    xt     = read_fvecs(f"{SIFT_DIR}/sift_learn.fvecs")
    print("✅ Loaded SIFT1M")
    print(f"   Base: {xb.shape} | Query: {xq.shape} | GT: {I_true.shape} | Learn: {xt.shape}")
    return xb, xq, I_true, xt

# ── Build index ───────────────────────────────────────────────────────────────

def build_index(xb):
    vectors = torch.from_numpy(xb)
    ids     = torch.arange(len(xb))

    build_params        = quake.IndexBuildParams()
    build_params.nlist  = NLIST
    build_params.metric = METRIC

    index = quake.QuakeIndex()
    index.build(vectors, ids, build_params)
    print("✅ Index built")
    return index

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    xb, xq, I_true, xt = load_sift1m()
    index = build_index(xb)