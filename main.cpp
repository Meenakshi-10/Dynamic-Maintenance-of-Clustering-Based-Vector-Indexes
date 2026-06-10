/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <fstream>

#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) && "unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double t0 = elapsed();

    const char* index_key  = "IVF4096,Flat";
    const size_t n_initial = 100,000; // vectors added before the growth loop
    const size_t batch_size = 100;   // vectors per incremental batch
    const size_t measure_every = 10; // measure recall every N batches
    const int    nprobe = 64;        // fixed search parameter throughout

    faiss::Index* index;
    size_t d;

    // ------------------------------------------------------------------ train
    //after this, we have 4096 centroids without any vectors assigned to them
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        size_t nt;
        float* xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0, index_key, d);
        index = faiss::index_factory(d, index_key);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
        index->train(nt, xt);
        delete[] xt;
    }

    // --------------------------------------------------------------- load base
    size_t nb;
    float* xb;
    {
        size_t d2;
        xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 && "base dimension mismatch");
        printf("[%.3f s] Loaded base set: %ld vectors\n", elapsed() - t0, nb);
    }

    // ---------------------------------------------------------- initial add()
    assert(n_initial <= nb && "n_initial exceeds base set size");
    printf("[%.3f s] Adding initial %ld vectors\n", elapsed() - t0, n_initial);
    index->add(n_initial, xb);

    // ---------------------------------------------------------- load queries
    size_t nq;
    float* xq;
    {
        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 && "query dimension mismatch");
        printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, nq);
    }

    // ------------------------------------------------------- load ground truth
    // Ground truth is against the full 1M base set, so recall will naturally
    // start low and rise as the true nearest neighbours are added to the index.
    size_t k;
    faiss::Index::idx_t* gt;
    {
        size_t nq2;
        int* gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq && "ground-truth query count mismatch");
        gt = new faiss::Index::idx_t[k * nq];
        for (size_t i = 0; i < k * nq; i++)
            gt[i] = gt_int[i];
        delete[] gt_int;
        printf("[%.3f s] Loaded ground truth (k=%ld)\n", elapsed() - t0, k);
    }

    // ------------------------------------------- fix nprobe for all searches
    {
        faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        assert(ivf && "expected an IVF index");
        ivf->nprobe = nprobe;
        printf("[%.3f s] nprobe set to %d\n", elapsed() - t0, nprobe);
    }

    // -------------------------------------------------- recall helper lambda
    std::ofstream metrics_file("recall_metrics.csv");
    metrics_file << "ntotal,R@1,R@10,R@100\n";

    faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * k];
    float*               D = new float[nq * k];

    auto measure_recall = [&]() -> std::tuple<float, float, float> {
        index->search(nq, xq, k, D, I);
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (size_t i = 0; i < nq; i++) {
            // R@r: any of the true top-r GT neighbors found in retrieved top-r
            auto hit = [&](size_t r) -> bool {
                for (size_t j = 0; j < r; j++)
                    for (size_t g = 0; g < r; g++)
                        if (I[i * k + j] == gt[i * k + g]) return true;
                return false;
            };
            if (hit(1))   n_1++;
            if (hit(10))  n_10++;
            if (hit(100)) n_100++;
        }
        return {n_1 / float(nq), n_10 / float(nq), n_100 / float(nq)};
    };

    // ----------------------------------------- recall at the initial snapshot
    {
        auto [r1, r10, r100] = measure_recall();
        printf("[%.3f s] ntotal=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f\n",
               elapsed() - t0, index->ntotal, r1, r10, r100);
        metrics_file << index->ntotal << "," << r1 << "," << r10 << "," << r100 << "\n";
    }

    // ---------------------------------------------- incremental growth loop
    size_t n_remaining = nb - n_initial; //900,000
    size_t n_batches   = n_remaining / batch_size; //9000

    printf("[%.3f s] Growing index: %ld batches x %ld vectors "
           "(measuring every %ld batches)\n",
           elapsed() - t0, n_batches, batch_size, measure_every);

    for (size_t b = 0; b < n_batches; b++) {
        size_t offset = n_initial + b * batch_size;
        index->add(batch_size, xb + offset * d);

        if ((b + 1) % measure_every == 0) {
            auto [r1, r10, r100] = measure_recall();
            printf("[%.3f s] ntotal=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f\n",
                   elapsed() - t0, index->ntotal, r1, r10, r100);
            metrics_file << index->ntotal << "," << r1 << "," << r10 << "," << r100 << "\n";
        }
    }

    // ---------------------------------------------------- flush any remainder
    size_t added = n_initial + n_batches * batch_size;
    if (added < nb) {
        index->add(nb - added, xb + added * d);
        auto [r1, r10, r100] = measure_recall();
        printf("[%.3f s] ntotal=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f  [final]\n",
               elapsed() - t0, index->ntotal, r1, r10, r100);
        metrics_file << index->ntotal << "," << r1 << "," << r10 << "," << r100 << "\n";
    }

    metrics_file.close();
    delete[] I;
    delete[] D;
    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
