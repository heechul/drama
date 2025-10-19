// Onebank attacker that targets single bank of the memory 

#include <bitset>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <map>
#include <list>
#include <utility>
#include <fstream>
#include <set>
#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>
#include <sstream>
#include <iterator>
#include <math.h>
#include <signal.h>
#include <sys/ioctl.h>
#include "measure.h"
#include <pthread.h>


#define POINTER_SIZE       (sizeof(void*) * 8) // #of bits of a pointer
#define MAX_XOR_BITS       15    // orig: 7
// ------------ global settings ----------------
int verbosity = 1;
size_t g_page_size;

int g_scale_factor = 1; // scale factor for timing (to adjust for different CPU speeds)
volatile int g_access_type = 0; // 0: read, 1: write
bool g_flush_cacheline = true; // true: flush, false: no flush

// default values
size_t num_reads_outer = 10;
#if defined(__aarch64__)
size_t num_reads_inner = 10; // need amplication due to low timer resolution in ARM64
#else
size_t num_reads_inner = 1;
#endif
size_t mapping_size = (1ULL<<30); // 1GB default
size_t expected_sets = 16;
int g_start_bit = 5; // search start bit
int g_end_bit = 40; // search end bit
char* g_output_file = nullptr;
volatile int g_quit_signal = 0;
// ----------------------------------------------

#define ETA_BUFFER 5
#define MAX_HIST_SIZE 2000

std::vector <std::vector<pointer>> sets;
std::map<int, std::vector<pointer> > functions;

int g_pagemap_fd = -1;
void *mapping;

// ----------------------------------------------
size_t getPhysicalMemorySize() {
    struct sysinfo info;
    sysinfo(&info);
    return (size_t) info.totalram * (size_t) info.mem_unit;
}

// ----------------------------------------------
const char *getCPUModel() {
    static char model[64];
    char *buffer = NULL;
    size_t n, idx;
    FILE *f = fopen("/proc/cpuinfo", "r");
    while (getline(&buffer, &n, f) > 0) {
        idx = 0;
        if (strncmp(buffer, "Model", 5) == 0 || 
            strncmp(buffer, "model name", 10) == 0) 
	    {
            while (buffer[idx] != ':')
                idx++;
            idx += 2;
            strcpy(model, &buffer[idx]);
            idx = 0;
            while (model[idx] != '\n')
                idx++;
            model[idx] = 0;
            break;
        }
    }
    fclose(f);
    return model;
}

// ----------------------------------------------
uint64_t phy_start_addr;

void setupMapping() {
 
    // try 1GB huge page
    mapping = mmap(NULL, mapping_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE |
                    (30 << MAP_HUGE_SHIFT), -1, 0);
    if ((void *)mapping == MAP_FAILED) {
        // try 2MB huge page
        mapping = mmap(NULL, mapping_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                        -1, 0);
        if ((void *)mapping == MAP_FAILED) {
            // nomal page allocation
            mapping = mmap(NULL, mapping_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
            if ((void *)mapping == MAP_FAILED) {
                perror("alloc failed");
                exit(1);
            } else
                logInfo("small page mapping (%zu KB)\n", g_page_size / 1024);
        } else
            logInfo("%s huge page mapping\n", "2MB");
    } else
        logInfo("%s huge page mapping\n", "1GB");

    assert(mapping != (void *) -1);

    logDebug("%s", "Initialize large memory block...\n");
    for (size_t index = 0; index < mapping_size; index += g_page_size) {
        pointer *temporary =
            reinterpret_cast<pointer *>(static_cast<uint8_t *>(mapping)
                                        + index);
        temporary[0] = index;
    }
    logDebug("%s", " done!\n");
}

// ----------------------------------------------
long utime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec) * 1000000 + (tv.tv_usec);
}


// ----------------------------------------------

#if defined(__aarch64__)
#define USE_TIMER_THREAD    0

static volatile uint64_t counter = 0;
static pthread_t count_thread;

static void *countthread(void *dummy) {
    uint64_t local_counter = 0;
    while (1) {
        local_counter++;
        counter = local_counter;
    }
    return NULL;
}
#endif

uint64_t rdtsc() {
#if defined(__aarch64__) && USE_TIMER_THREAD==1
    asm volatile ("DSB SY");	
    return counter;
#elif defined(__aarch64__) && USE_TIMER_THREAD==0
    uint64_t virtual_timer_value;
    asm volatile("isb");
    asm volatile("mrs %0, cntvct_el0" : "=r" (virtual_timer_value));
    return virtual_timer_value;
#else    
    uint64_t a, d;
    asm volatile ("xor %%rax, %%rax\n" "cpuid"::: "rax", "rbx", "rcx", "rdx");
    asm volatile ("rdtscp" : "=a" (a), "=d" (d) : : "rcx");
    a = (d << 32) | a;
    return a;
#endif
}

// ----------------------------------------------
uint64_t rdtsc2() {
#if defined(__aarch64__)
    return rdtsc();
#else
    uint64_t a, d;
    asm volatile ("rdtscp" : "=a" (a), "=d" (d) : : "rcx");
    asm volatile ("cpuid"::: "rax", "rbx", "rcx", "rdx");
    a = (d << 32) | a;
    return a;
#endif
}


static inline void clflush(volatile void *p) {
#if defined(__aarch64__)
    asm volatile("DC CIVAC, %[ad]" : : [ad] "r" (p) : "memory");
#else
    asm volatile("clflush (%0)" : : "r" (p) : "memory");
#endif
}

static inline void clflushopt(volatile void *p) {
#if defined(__aarch64__)
    asm volatile("DC CIVAC, %[ad]" : : [ad] "r" (p) : "memory");
#else
    asm volatile("clflushopt (%0)" : : "r" (p) : "memory");
#endif
}

static inline void clwb(volatile void *p) {
#if defined(__aarch64__)
    asm volatile("DC CIVAC, %[ad]" : : [ad] "r" (p) : "memory");
#else
    asm volatile("clwb (%0)" : : "r" (p) : "memory");
#endif
}

static inline void sfence() {
#if defined(__aarch64__)
    asm volatile("DSB SY");
#else
    asm volatile("sfence" ::: "memory");
#endif
}


// ----------------------------------------------
uint64_t getTiming(pointer first, pointer second) {
    size_t min_res = (-1ull);

    for (int i = 0; i < num_reads_outer; i++) {
        size_t number_of_reads = num_reads_inner;
        volatile size_t *f = (volatile size_t *) first;
        volatile size_t *s = (volatile size_t *) second;

        *f;
        *s;
        clflush(f);
        clflush(s);

        size_t t0 = rdtsc();

        while (number_of_reads-- > 0) {
            *f;
            *s;
            clflush(f);
            clflush(s);
        }

        uint64_t res = (rdtsc2() - t0);
        res = res / g_scale_factor;
        if (res < min_res)
            min_res = res;
    }
    return min_res;
}

// ----------------------------------------------
void getRandomAddress(pointer *virt) {
    size_t unit_size = (1ULL << g_start_bit);
    size_t offset = (size_t)(rand() % (mapping_size / unit_size)) * unit_size;
    *virt = (pointer) mapping + offset;
}

// Worker thread argument
struct ThreadArg {
    int id;
    long *counter;
    std::vector<pointer> *local_set;
};

// Each thread repeatedly accesses all addresses in sets[0] until g_quit_signal
// is set. It increments its own counter for each full traversal of the set.
static void *access_all_thread(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    long *ctr = a->counter;
    int cpu_id = a->id; // bind thread to core with same id as thread id

    logInfo("Setting CPU affinity to core %d\n", cpu_id);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) != 0) {
        perror("sched_setaffinity");
        exit(1);
    }

    // create a local copy of sets[0] to avoid pointer chasing and improve locality
    std::vector<pointer> *local_set = a->local_set;

    logInfo("Thread %d: accessing %ld addresses in a loop...\n", cpu_id, (long)local_set->size());

    // prepare raw pointer + size for tight loops (avoids repeated bounds checks)
    pointer *data = local_set->empty() ? nullptr : local_set->data();
    size_t n = local_set->size();

    // main loop
    int cur_access = g_access_type;
    while (!g_quit_signal) {
        if (cur_access != g_access_type) {
            cur_access = g_access_type;
            logInfo("Thread %d: switching access type to %s\n",
                    cpu_id, (cur_access == 1) ? "write" : "read");
            *ctr = 0; // reset counter
        }
        if (cur_access == 1) {
            // write attack
            for (size_t j = 0; j < n; ++j) {
                // write to the address and flush it
                if (g_flush_cacheline) clflushopt((void *)data[j]);
                *((volatile int *)data[j]) = 0xdeadbeef;
            }
        } else {
            // read attack
            for (size_t j = 0; j < n; ++j) {
                // touch the address and flush it
                *((volatile int *)data[j]);
                if (g_flush_cacheline) clflushopt((void *)data[j]);
            }
        }
        (*ctr)++;
    }
    return NULL;
}

// name_bits: return a human-readable string listing the bit positions
// set in `mask`. Example: mask with bits 5 and 7 => "5 7". Used for
// printing and saving discovered functions.


// ----------------------------------------------
int main(int argc, char *argv[]) {
    size_t tries, t;
    std::set <pointer> addr_pool;
    std::map<int, std::list<pointer> > timing;
    size_t hist[MAX_HIST_SIZE];
    int c;
    int samebank_threshold = -1;
    int cpu_affinity = 0;
    int target_n = 125; // target number of addresses per set
    int num_threads = 1;

    // parse command line arguments
    while ((c = getopt(argc, argv, "a:b:c:e:r:g:m:i:j:k:s:t:v:f:n:")) != EOF) {
        switch (c) {
        case 'a':
            g_access_type = (!strncmp(optarg, "write", 5)) ? 1 : 0;
            break;
        case 'b':
            g_start_bit = atoi(optarg);
            break;
        case 'e':
            g_end_bit = atoi(optarg);
            break;
        case 'c':
            cpu_affinity = atoi(optarg);
            break;
        case 'g':
            mapping_size = atol(optarg) * 1024 * 1024 * 1024;
            break;
        case 'm':
            mapping_size = atol(optarg) * 1024 * 1024;
            break;
        case 'r':
            g_scale_factor = atoi(optarg);
            break;
        case 'i':
            num_reads_outer = atol(optarg);
            break;
        case 'j':
            num_reads_inner = atol(optarg);
            break;
        case 'k':
            target_n = atoi(optarg);
            break;
        case 's':
            expected_sets = atoi(optarg);
            break;
        case 't':
            samebank_threshold = atoi(optarg);
            break;
        case 'v':
            verbosity = atoi(optarg);
            break;
        case 'f':
            g_flush_cacheline = (!strncmp(optarg, "noflush", 7)) ? false : true;
            break;
        case 'n':
            num_threads = atoi(optarg);
            if (num_threads <= 0) num_threads = 1;
            break;
        case ':':
            printf("Missing option.\n");
            exit(1);
            break;
        default:
            printf(
                "Usage %s [-m <memory size in MB> | -g <memory size in GB>] [-i <number of outer loops>] [-j <number of inner loops>] [-k <target addresses per set>] [-s <expected sets>] [-t <threshold cycles>] [-f <output file>]\n",
                argv[0]);
            exit(0);
            break;
        }
    }

    srand(time(NULL));
    g_page_size = sysconf(_SC_PAGESIZE);
    setupMapping();

    logInfo("Mapping has %zu MB\n", mapping_size / 1024 / 1024);

    logDebug("CPU: %s\n", getCPUModel());
    logDebug("Number of reads: %lu x %lu\n", num_reads_outer, num_reads_inner)
    logDebug("Expected sets: %lu\n", expected_sets);

    pointer first, second;
    pointer base;

    int found_sets = 0;
    int found_siblings = 0;

    tries = expected_sets * target_n; // DEBUG: original 125.

    // build address pool
    for (int i = 0; i < tries; i++) {
        getRandomAddress(&second);
        addr_pool.insert(second);
    }

    auto ait = addr_pool.begin();
    // std::advance(ait, rand() % addr_pool.size());
    base = *ait;

    logDebug("Address pool size: %lu\n", addr_pool.size());

#if defined(__aarch64__) && USE_TIMER_THREAD==1
    int rr = pthread_create(&count_thread, 0, countthread , 0);
    if (rr != 0) {
        return -1;
    }
    logDebug("%s\n", "Waiting the counter thread...");
    while(counter == 0) {
        asm volatile("DSB SY");
    }
    logDebug("Done: %ld\n", counter);    
#endif

    // if SIGINT is received, set g_quit_signal to 1
    signal(SIGINT, [](int signum) {
        g_quit_signal = 1;
    });

    // row hit timing
    t = getTiming(base, base + 64);
    logInfo("Average ROW hit cycles: %ld \n", t);

    int failed;
    uint64_t measure_count = 0;

    while (found_sets == 0) {
        for (size_t i = 0; i < MAX_HIST_SIZE; ++i)
            hist[i] = 0;
        failed = 0;
    search_set:
        failed++;
        if (failed > 10) {
            logWarning("%s\n", "Couldn't find set after 10 tries, giving up, sorry!");
            break;
        }

        // choose base address from remaining addresses
        auto ait = addr_pool.begin();
        std::advance(ait, rand() % addr_pool.size());
        base = *ait;
        addr_pool.erase(ait);

        logInfo("Searching for set %d (try %d)\n", found_sets + 1, failed);
        timing.clear();

        auto pool_it = addr_pool.begin();
        // iterate over the addr_pool and measure access times
        measure_count = 0;
        while (pool_it != addr_pool.end()) {

            first = *pool_it;

            // measure timing
            t = getTiming(base, first);
            measure_count++;

            // sched_yield();
            timing[t].push_back(first);

            // advance iterator
            pool_it++;
        }
        printf("(%lu)\n", measure_count);

        // identify sets -> must be on the right, separated in the histogram
        std::vector <pointer> new_set;
        std::map < int, std::list < pointer > > ::iterator hit;
        int min = MAX_HIST_SIZE;
        int max = 0;
        int max_v = 0;
        for (hit = timing.begin(); hit != timing.end(); hit++) {
	        assert(hit->first < MAX_HIST_SIZE);
            hist[hit->first] = hit->second.size();
            if (hit->first > max)
                max = hit->first;
            if (hit->first < min)
                min = hit->first;
            if (hit->second.size() > max_v)
                max_v = hit->second.size();
        }
        logDebug("Timing histogram: min: %d max: %d max_v: %d\n", min, max, max_v);

        // scale histogram
        double scale_v = (double) (100.0)
                         / (max_v > 0 ? (double) max_v : 100.0);
        assert(scale_v >= 0);
        // while (hist[++min] <= 1);
        // while (hist[--max] <= 1);

        // print histogram
        for (size_t i = min; i <= max; i++) {
            printf("%03zu: %4zu ", i, hist[i]);
            assert(hist[i] >= 0);
            for (size_t j = 0; j < hist[i] * scale_v && j < 80; j++) {
                printf("#");
            }

            printf("\n");
        }
        // find separation
        int empty = 0, found = 0;

        if (samebank_threshold > 0) {
            found = samebank_threshold;
        } else {
            if (samebank_threshold == -2) {
                // weighted k-means for 2 clusters using hist[] as weight
                double cluster1 = (double)min;
                double cluster2 = (double)max;
                double prev_cluster1 = -1e9;
                double prev_cluster2 = -1e9;
                int max_iterations = 1000;
                int iterations = 0;
                const double EPS = 1e-6;

                while ((fabs(cluster1 - prev_cluster1) > EPS || fabs(cluster2 - prev_cluster2) > EPS) &&
                       iterations < max_iterations) {
                    prev_cluster1 = cluster1;
                    prev_cluster2 = cluster2;

                    int64_t sum1 = 0, sum2 = 0;
                    int64_t cnt1 = 0, cnt2 = 0;

                    for (int b = min; b <= max; b++) {
                        size_t c = hist[b];
                        if (c == 0) continue;
                        int d1 = abs(b - cluster1);
                        int d2 = abs(b - cluster2);
                        if (d1 < d2) {
                            sum1 += c * b;
                            cnt1 += c;
                        } else {
                            sum2 += c * b;
                            cnt2 += c;
                        }
                    }

                    if (cnt1 > 0) cluster1 = (double)sum1 / cnt1;
                    if (cnt2 > 0) cluster2 = (double)sum2 / cnt2;

                    iterations++;

                    logDebug("K-means iteration %d: cluster1: %.2f cluster2: %.2f\n",
                             iterations, cluster1, cluster2);
                }

                found = (int)(cluster2 - (cluster2 - cluster1) / 4.0); // biased towards cluster2
                logDebug("K-means clustering found threshold at %d (cluster1: %d, cluster2: %d)\n",
                         found, (int)cluster1, (int)cluster2);
            } else {
                // find a gap of at least 5 empty bins, starting from the right (high cycle counts)
                for (int i = max; i >= min; i--) {
                    if (hist[i] <= 1)
                        empty++;
                    else
                        empty = 0;
                    if (empty >= 5) {
                        found = i + empty;
                        break;
                    }
                }
            }

            if (!found) {
                logWarning("%s\n", "No set found, trying again...");
                goto search_set;
            }
        }

        new_set.push_back(base); // this is needed. another bug in the original code

        // add all addresses with timing >= found && <= max (= row conflict)
        for (hit = timing.begin(); hit != timing.end(); hit++) {
            if (hit->first >= found && hit->first <= max) {
                for (std::list<pointer>::iterator it = hit->second.begin();
                     it != hit->second.end(); it++) {
                    new_set.push_back(*it);
                }
            }
        }

	    logInfo("found(cycles): %d newset_sz: %lu (expected_sz: %lu) pool_sz: %lu\n",
                 found, new_set.size(), tries/expected_sets, addr_pool.size());

        if (new_set.size() <= tries / expected_sets * 0.1) {
            logWarning("Set must be wrong, contains too few addresses (%lu). Try again...\n", new_set.size());
            goto search_set;
        }
        if (new_set.size() > tries / expected_sets * 2) {
	        logWarning("Set must be wrong, contains too many addresses (expected: %lu/found: %ld). Try again...\n", tries / expected_sets, new_set.size());
            goto search_set;
        }

        // validate if all addresses in the new set are indeed same-bank
        logDebug("Validating set with %lu addresses...\n", new_set.size());
        for (size_t i = 1; i < new_set.size(); i++) {
            // re-measure timing (exclude base address at index 0)
            t = getTiming(base, new_set[i]);
            if (t < found) {
                logWarning("Validation failed: address 0x%lx has timing %lu < %d. removing it from the set\n",
                           new_set[i], t, found);
                // remove it from the new set
                new_set.erase(new_set.begin() + i);
                i--;
            }
        }
        logDebug("Validation done. New set size: %lu\n", new_set.size());

        // save identified set if one was found
        sets.push_back(new_set);
        found_siblings += new_set.size();

        found_sets++;
    }
    logDebug("Done measuring. found_sets: %d found_siblings: %d\n",
             found_sets, found_siblings);

    
    // access all addresses in the sets[0] using multiple threads
    if (sets.empty()) {
        logWarning("%s\n", "No sets found, nothing to access");
        exit(1);
    }

    printf("Accessing (%s) all addresses in the sets[0] (%ld addresses) with %d threads...\n",
        (g_access_type==1)?"write":"read",
        (long)sets[0].size(), num_threads);

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArg> args(num_threads);
    std::vector<long> counters(num_threads, 0);
    std::vector<std::vector<pointer>> local_sets(num_threads);

    // divide sets[0] into num_threads local sets
    for (int i = 0; i < num_threads; ++i) {
        local_sets[i].clear();
    }
    for (size_t i = 0; i < sets[0].size(); ++i) {
        local_sets[i % num_threads].push_back(sets[0][i]);
    }

    long t0 = utime();

    for (int i = 0; i < num_threads; ++i) {
        args[i].id = cpu_affinity + i;
        args[i].counter = &counters[i];
        args[i].local_set = &local_sets[i];

        int rc = pthread_create(&threads[i], NULL, access_all_thread, &args[i]);
        if (rc != 0) {
            perror("pthread_create");
            // continue creating remaining threads or exit? we'll exit
            g_quit_signal = 1;
            break;
        }
    }

    // if SIGUSR1 is received, set g_access_type to 0 (read)
    signal(SIGUSR1, [](int signum) {
        g_access_type = 0;
    });

    // if SIGUSR2 is received, set g_access_type to 1 (write)
    signal(SIGUSR2, [](int signum) {
        g_access_type = 1;
    });

    // main thread just waits for SIGINT to set g_quit_signal
    int cur_access = g_access_type;
    while (!g_quit_signal) {
        if (cur_access != g_access_type) {
            cur_access = g_access_type;
            printf("Main thread: switching access type to %s\n",
                    (cur_access == 1) ? "write" : "read");
            t0 = utime();
        }
        sleep(1);
    }

    // join threads and aggregate counters
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    long dur_in_us = utime() - t0;
    long long accessed_bytes = 0;
    // per-thread b/w
    for (int i = 0; i < num_threads; ++i) {
        long long t_bytes = (long long)local_sets[i].size() * counters[i] * 64LL;
        if (dur_in_us > 0) {
            double mbps = (double)t_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
            printf("Thread %d: iters: %ld  Bandwidth: %.1f MB/s\n", i, counters[i], mbps);
        } else {
            printf("Thread %d: iters: %ld  Bandwidth: infinite (duration 0)\n", i, counters[i]);
        }
        accessed_bytes += t_bytes;
    }
    // total b/w
    if (dur_in_us > 0) {
        double total_mbps = (double)accessed_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
        printf("Total aggregate bandwidth: %.1f MB/s\n", total_mbps);
    } else {
        printf("Total aggregate bandwidth: infinite (duration 0)\n");
    }
    exit(1);
    return 0;
}
