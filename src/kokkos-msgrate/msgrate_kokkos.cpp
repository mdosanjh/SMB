/* -*- C -*-
 *
 * Copyright 2006 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
 * retains certain rights in this software.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, 
 * Boston, MA  02110-1301, USA.
 */


/* -*- C++ -*-
 *
 * Kokkos-converted msgrate benchmark.
 */

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cstddef>
#include <cstdint>

/*
 * Buffer memory space.
 *
 * If Kokkos is built with CUDA/HIP and the default execution space is CUDA/HIP,
 * these buffers will be allocated in device memory. This requires GPU-aware MPI.
 *
 * If you do not have GPU-aware MPI, change this to Kokkos::HostSpace.
 */
using ExecSpace = Kokkos::DefaultExecutionSpace;
using BufferMemorySpace = ExecSpace::memory_space;

using UCharDeviceView =
    Kokkos::View<unsigned char*, BufferMemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using ConstUCharDeviceView =
    Kokkos::View<const unsigned char*, BufferMemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using UCharHostView =
    Kokkos::View<unsigned char*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using ConstUCharHostView =
    Kokkos::View<const unsigned char*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

static void
abort_app(const char *msg)
{
    perror(msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/**
 * Kokkos-aware helper functions
 */
static void* bench_alloc(size_t size);
static void  bench_free(void* ptr);
static void  bench_memcpy_to_host(void* dst, const void* src, size_t size);
static void  bench_memcpy_from_host(void* dst, const void* src, size_t size);
static void  bench_sync(void);
static void  bench_cleanup(void);

static void*
bench_alloc(size_t size)
{
    void* p = Kokkos::kokkos_malloc<BufferMemorySpace>("bench_buffer", size);

    if (p == nullptr) {
        fprintf(stderr, "Kokkos allocation failed for %zu bytes\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    UCharDeviceView v(static_cast<unsigned char*>(p), size);
    Kokkos::deep_copy(v, static_cast<unsigned char>(0));
    Kokkos::fence();

    return p;
}

static void
bench_free(void* ptr)
{
    if (ptr == nullptr) return;

    Kokkos::kokkos_free<BufferMemorySpace>(ptr);
}

static void
bench_sync(void)
{
    Kokkos::fence();
}

static void
bench_cleanup(void)
{
    Kokkos::fence();
}

static void
bench_memcpy_to_host(void* dst, const void* src, size_t size)
{
    UCharHostView h_dst(static_cast<unsigned char*>(dst), size);
    ConstUCharDeviceView d_src(static_cast<const unsigned char*>(src), size);

    Kokkos::deep_copy(h_dst, d_src);
    Kokkos::fence();
}

static void
bench_memcpy_from_host(void* dst, const void* src, size_t size)
{
    UCharDeviceView d_dst(static_cast<unsigned char*>(dst), size);
    ConstUCharHostView h_src(static_cast<const unsigned char*>(src), size);

    Kokkos::deep_copy(d_dst, h_src);
    Kokkos::fence();
}

/** Verification helper functions */
KOKKOS_INLINE_FUNCTION
static unsigned char
pattern_byte(int src_rank, int msg_idx, size_t byte_idx)
{
    unsigned int x = static_cast<unsigned int>(src_rank);
    x = x * 131u + static_cast<unsigned int>(msg_idx);
    x = x * 131u + static_cast<unsigned int>(byte_idx);
    return static_cast<unsigned char>(x & 0xffu);
}

static void
init_send_buffer(char* buf, int rank, int npeers, int nmsgs, size_t nbytes)
{
    const size_t total =
        static_cast<size_t>(npeers) * static_cast<size_t>(nmsgs) * nbytes;

    auto policy =
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(0, total);

    Kokkos::parallel_for(
        "init_send_buffer",
        policy,
        KOKKOS_LAMBDA(const size_t idx) {
            const size_t b = idx % nbytes;
            const size_t msg_linear = idx / nbytes;
            const int k = static_cast<int>(msg_linear % static_cast<size_t>(nmsgs));

            reinterpret_cast<unsigned char*>(buf)[idx] =
                pattern_byte(rank, k, b);
        });

    Kokkos::fence();
}

static void
verify_recv_buffer(char* buf,
                   int rank,
                   int* recv_peers,
                   int npeers,
                   int nmsgs,
                   size_t nbytes)
{
    const size_t total =
        static_cast<size_t>(npeers) * static_cast<size_t>(nmsgs) * nbytes;

    unsigned char* host = static_cast<unsigned char*>(std::malloc(total));
    if (host == nullptr) abort_app("malloc");

    bench_memcpy_to_host(host, buf, total);
    bench_sync();

    for (int j = 0; j < npeers; ++j) {
        const int src_rank = recv_peers[j];

        for (int k = 0; k < nmsgs; ++k) {
            const size_t base =
                (static_cast<size_t>(j) * static_cast<size_t>(nmsgs) +
                 static_cast<size_t>(k)) *
                nbytes;

            for (size_t b = 0; b < nbytes; ++b) {
                const unsigned char expected = pattern_byte(src_rank, k, b);

                if (host[base + b] != expected) {
                    fprintf(stderr,
                            "Verification failed on rank %d at recv_slot=%d "
                            "msg=%d byte=%zu: got=%u expected=%u src=%d\n",
                            rank,
                            j,
                            k,
                            b,
                            static_cast<unsigned int>(host[base + b]),
                            static_cast<unsigned int>(expected),
                            src_rank);

                    std::free(host);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
    }

    std::free(host);
}

/* constants */
const int magic_tag = 1;

/* configuration parameters - setable by command line arguments */
int npeers = 6;
int niters = 4096;
int nmsgs = 128;
size_t nbytes = 8;
size_t cache_size = 8 * 1024 * 1024 / sizeof(int);
int ppn = -1;
int machine_output = 0;
int verify = 0;

/* globals */
int *send_peers = nullptr;
int *recv_peers = nullptr;
int *cache_buf = nullptr;
char *send_buf = nullptr;
char *recv_buf = nullptr;
MPI_Request *reqs = nullptr;

int rank = -1;
int world_size = -1;

static void
cache_invalidate(void)
{
    /*
     * Original CUDA/HIP version skipped cache invalidation for GPU buffers.
     * For Kokkos, this remains a host-side cache trashing operation.
     */
    if constexpr (std::is_same_v<BufferMemorySpace, Kokkos::HostSpace>) {
        cache_buf[0] = 1;

        for (size_t i = 1; i < cache_size; ++i) {
            cache_buf[i] = cache_buf[i - 1];
        }
    } else {
        return;
    }
}

static inline double
timer(void)
{
    return MPI_Wtime();
}

void
display_result(const char *test, const double result)
{
    if (0 == rank) {
        if (machine_output) {
            printf("%.2f ", result);
        } else {
            printf("%10s: %.2f\n", test, result);
        }
    }
}

void
test_one_way(void)
{
    int i, k, nreqs;
    double tmp, total = 0.0;
    MPI_Comm comm;

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_size % 2 == 1) {
        MPI_Comm_split(MPI_COMM_WORLD,
                       (rank == world_size - 1) ? MPI_UNDEFINED : 1,
                       rank,
                       &comm);
    } else {
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    }

    if (!(world_size % 2 == 1 && rank == (world_size - 1))) {
        if (rank < world_size / 2) {
            for (i = 0; i < niters; ++i) {
                cache_invalidate();

                MPI_Barrier(comm);

                tmp = timer();
                nreqs = 0;

                for (k = 0; k < nmsgs; ++k) {
                    MPI_Isend(send_buf + nbytes * k,
                              static_cast<int>(nbytes),
                              MPI_CHAR,
                              rank + world_size / 2,
                              magic_tag,
                              comm,
                              &reqs[nreqs++]);
                }

                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
                total += timer() - tmp;
            }
        } else {
            for (i = 0; i < niters; ++i) {
                cache_invalidate();

                MPI_Barrier(comm);

                tmp = timer();
                nreqs = 0;

                for (k = 0; k < nmsgs; ++k) {
                    MPI_Irecv(recv_buf + nbytes * k,
                              static_cast<int>(nbytes),
                              MPI_CHAR,
                              rank - world_size / 2,
                              magic_tag,
                              comm,
                              &reqs[nreqs++]);
                }

                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
                total += timer() - tmp;
            }
        }

        MPI_Allreduce(&total, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm);
        display_result("single direction",
                       (niters * nmsgs) / (tmp / world_size));

        MPI_Comm_free(&comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void
test_same_direction(void)
{
    int i, j, k, nreqs;
    double tmp, total = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < niters; ++i) {
        cache_invalidate();

        MPI_Barrier(MPI_COMM_WORLD);

        tmp = timer();

        for (j = 0; j < npeers; ++j) {
            nreqs = 0;

            for (k = 0; k < nmsgs; ++k) {
                MPI_Irecv(recv_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          recv_peers[j],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }

            for (k = 0; k < nmsgs; ++k) {
                MPI_Isend(send_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          send_peers[npeers - j - 1],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }

            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
        }

        total += timer() - tmp;
    }

    MPI_Allreduce(&total, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    display_result("pair-based",
                   (niters * npeers * nmsgs * 2) / (tmp / world_size));
}

void
test_prepost(void)
{
    int i, j, k, nreqs = 0;
    double tmp, total = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    tmp = timer();

    for (j = 0; j < npeers; ++j) {
        for (k = 0; k < nmsgs; ++k) {
            MPI_Irecv(recv_buf + nbytes * (k + j * nmsgs),
                      static_cast<int>(nbytes),
                      MPI_CHAR,
                      recv_peers[j],
                      magic_tag,
                      MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }
    }

    total += timer() - tmp;

    for (i = 0; i < niters - 1; ++i) {
        cache_invalidate();

        MPI_Barrier(MPI_COMM_WORLD);

        tmp = timer();

        for (j = 0; j < npeers; ++j) {
            for (k = 0; k < nmsgs; ++k) {
                MPI_Isend(send_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          send_peers[npeers - j - 1],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }
        }

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        nreqs = 0;

        for (j = 0; j < npeers; ++j) {
            for (k = 0; k < nmsgs; ++k) {
                MPI_Irecv(recv_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          recv_peers[j],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }
        }

        total += timer() - tmp;
    }

    tmp = timer();

    for (j = 0; j < npeers; ++j) {
        for (k = 0; k < nmsgs; ++k) {
            MPI_Isend(send_buf + nbytes * (k + j * nmsgs),
                      static_cast<int>(nbytes),
                      MPI_CHAR,
                      send_peers[npeers - j - 1],
                      magic_tag,
                      MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }
    }

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    total += timer() - tmp;

    MPI_Allreduce(&total, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    display_result("pre-post",
                   (niters * npeers * nmsgs * 2) / (tmp / world_size));
}

void
test_allstart(void)
{
    int i, j, k, nreqs = 0;
    double tmp, total = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < niters; ++i) {
        cache_invalidate();

        MPI_Barrier(MPI_COMM_WORLD);

        tmp = timer();
        nreqs = 0;

        for (j = 0; j < npeers; ++j) {
            for (k = 0; k < nmsgs; ++k) {
                MPI_Irecv(recv_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          recv_peers[j],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }

            for (k = 0; k < nmsgs; ++k) {
                MPI_Isend(send_buf + nbytes * (k + j * nmsgs),
                          static_cast<int>(nbytes),
                          MPI_CHAR,
                          send_peers[npeers - j - 1],
                          magic_tag,
                          MPI_COMM_WORLD,
                          &reqs[nreqs++]);
            }
        }

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
        total += timer() - tmp;
    }

    MPI_Allreduce(&total, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    display_result("all-start",
                   (niters * npeers * nmsgs * 2) / (tmp / world_size));
}

void
usage(void)
{
    fprintf(stderr, "Usage: msgrate_kokkos -n <ppn> [OPTION]...\n\n");
    fprintf(stderr, "  -h           Display this help message and exit\n");
    fprintf(stderr, "  -p <num>     Number of peers used in communication\n");
    fprintf(stderr, "  -i <num>     Number of iterations per test\n");
    fprintf(stderr, "  -m <num>     Number of messages per peer per iteration\n");
    fprintf(stderr, "  -s <size>    Number of bytes per message\n");
    fprintf(stderr, "  -c <size>    Cache size in bytes\n");
    fprintf(stderr, "  -n <ppn>     Number of procs per node\n");
    fprintf(stderr, "  -o           Format output to be machine readable\n");
    fprintf(stderr, "  -v           Verify message contents after all-start\n");
    fprintf(stderr, "\n");
}

static void
bcast_size_t(size_t* value, int root, MPI_Comm comm)
{
    unsigned long long tmp = 0;

    if (rank == root) {
        tmp = static_cast<unsigned long long>(*value);
    }

    MPI_Bcast(&tmp, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
    *value = static_cast<size_t>(tmp);
}

int
main(int argc, char *argv[])
{
    int start_err = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /*
     * Kokkos initialization.
     *
     * This uses LOCAL_RANK as a device id if present.
     * If your system has more MPI ranks per node than GPUs, you may prefer to
     * control device binding externally via launcher options or Kokkos runtime
     * options.
     */
    {
        Kokkos::InitializationSettings settings;

        const char* local_rank = std::getenv("LOCAL_RANK");
        if (local_rank != nullptr) {
            settings.set_device_id(std::atoi(local_rank));
        }

        Kokkos::initialize(settings);
    }

    if (0 == rank) {
        int ch;

        while (start_err != 1 &&
               (ch = getopt(argc, argv, "p:i:m:s:c:n:ovh")) != -1) {
            switch (ch) {
            case 'p':
                npeers = std::atoi(optarg);
                break;
            case 'i':
                niters = std::atoi(optarg);
                break;
            case 'm':
                nmsgs = std::atoi(optarg);
                break;
            case 's':
                nbytes = static_cast<size_t>(std::strtoull(optarg, nullptr, 10));
                break;
            case 'c':
                cache_size =
                    static_cast<size_t>(std::strtoull(optarg, nullptr, 10)) /
                    sizeof(int);
                break;
            case 'n':
                ppn = std::atoi(optarg);
                break;
            case 'o':
                machine_output = 1;
                break;
            case 'v':
                verify = 1;
                break;
            case 'h':
            case '?':
            default:
                start_err = 1;
                usage();
            }
        }

        if (start_err != 1) {
            if (world_size < 3) {
                fprintf(stderr, "Error: At least three processes are required\n");
                start_err = 1;
            } else if (world_size <= npeers) {
                fprintf(stderr,
                        "Error: job size (%d) <= number of peers (%d)\n",
                        world_size,
                        npeers);
                start_err = 1;
            } else if (ppn < 1) {
                fprintf(stderr,
                        "Error: must specify processes per node using -n #\n");
                start_err = 1;
            } else if (world_size / ppn <= npeers) {
                fprintf(stderr, "Error: node count <= number of peers\n");
                start_err = 1;
            }

            if (nbytes > static_cast<size_t>(INT32_MAX)) {
                fprintf(stderr,
                        "Error: nbytes must fit in an MPI int count for this code\n");
                start_err = 1;
            }
        }
    }

    MPI_Bcast(&start_err, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (0 != start_err) {
        Kokkos::finalize();
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&npeers, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&niters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nmsgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bcast_size_t(&nbytes, 0, MPI_COMM_WORLD);
    bcast_size_t(&cache_size, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ppn, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&verify, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (0 == rank) {
        if (!machine_output) {
            printf("job size:   %d\n", world_size);
            printf("npeers:     %d\n", npeers);
            printf("niters:     %d\n", niters);
            printf("nmsgs:      %d\n", nmsgs);
            printf("nbytes:     %zu\n", nbytes);
            printf("cache size: %zu\n", cache_size * sizeof(int));
            printf("ppn:        %d\n", ppn);
            printf("Kokkos execution space: %s\n", ExecSpace::name());
        } else {
            printf("%d %d %d %d %zu %zu %d ",
                   world_size,
                   npeers,
                   niters,
                   nmsgs,
                   nbytes,
                   cache_size * sizeof(int),
                   ppn);
        }
    }

    const size_t buffer_size =
        static_cast<size_t>(npeers) * static_cast<size_t>(nmsgs) * nbytes;

    send_peers = static_cast<int*>(std::malloc(sizeof(int) * npeers));
    if (send_peers == nullptr) abort_app("malloc");

    recv_peers = static_cast<int*>(std::malloc(sizeof(int) * npeers));
    if (recv_peers == nullptr) abort_app("malloc");

    cache_buf = static_cast<int*>(std::malloc(sizeof(int) * cache_size));
    if (cache_buf == nullptr) abort_app("malloc");

    send_buf = static_cast<char*>(bench_alloc(buffer_size));
    recv_buf = static_cast<char*>(bench_alloc(buffer_size));

    reqs = static_cast<MPI_Request*>(
        std::malloc(sizeof(MPI_Request) * 2 * nmsgs * npeers));
    if (reqs == nullptr) abort_app("malloc");

    /*
     * Calculate peers.
     */
    for (int i = 0; i < npeers; ++i) {
        if (i < npeers / 2) {
            send_peers[i] =
                (rank + world_size + ((i - npeers / 2) * ppn)) % world_size;
        } else {
            send_peers[i] =
                (rank + world_size + ((i - npeers / 2 + 1) * ppn)) % world_size;
        }
    }

    if (npeers % 2 == 0) {
        for (int i = 0; i < npeers; ++i) {
            if (i < npeers / 2) {
                recv_peers[i] =
                    (rank + world_size + ((i - npeers / 2) * ppn)) % world_size;
            } else {
                recv_peers[i] =
                    (rank + world_size + ((i - npeers / 2 + 1) * ppn)) %
                    world_size;
            }
        }
    } else {
        for (int i = 0; i < npeers; ++i) {
            if (i < npeers / 2 + 1) {
                recv_peers[i] =
                    (rank + world_size + ((i - npeers / 2 - 1) * ppn)) %
                    world_size;
            } else {
                recv_peers[i] =
                    (rank + world_size + ((i - npeers / 2) * ppn)) % world_size;
            }
        }
    }

    if (verify) {
        init_send_buffer(send_buf, rank, npeers, nmsgs, nbytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    test_one_way();
    test_same_direction();
    test_prepost();
    test_allstart();

    if (verify) {
        verify_recv_buffer(recv_buf, rank, recv_peers, npeers, nmsgs, nbytes);
    }

    if (rank == 0 && machine_output) {
        printf("\n");
    }

    std::free(send_peers);
    std::free(recv_peers);
    std::free(cache_buf);
    std::free(reqs);

    bench_free(send_buf);
    bench_free(recv_buf);

    bench_cleanup();

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

