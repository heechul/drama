// A DRAM bank-aware memory performance attacker, which can target a subset DRAM banks
//
// Author: Heechul Yun (heechul.yun@ku.edu)
//
// Credit: heavily borrowed from DRAMA by Pessl et al. (USENIX Security 2016).
//
// Compile:
//    g++ -O2 -std=c++17 -o onebank_attacker onebank_attacker.cpp measure.cpp -lpthread
// Usage:
//   sudo ./onebank_attacker -m <memory size in MB> -s <expected sets> -l <target sets> -k <target addresses per set> -t <same-bank threshold cycles> -f <cache mode> -a <access type> -n <number of threads>
// Example:
//   sudo ./onebank_attacker -m 1024 -s 16 -l 1 -k 125 -t 300 -f 1 -a write -n 4
//     - This example allocates 1GB memory, searches for 16 DRAM sets, selects 1 set to attack, with 125 addresses in the set,
//       using 300 cycles as same-bank threshold, using write+clwb access mode, and 4 threads to access the addresses in the set.
// Note:
//   Adjust parameters as needed for your system.


#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <map>
#include <list>
#include <set>
#include <algorithm>
#include <sys/time.h>
#include <math.h>
#include <signal.h>
#include "measure.h"
#include <pthread.h>

// ------------ global settings ----------------
int verbosity = 1;
size_t g_page_size;

int g_scale_factor = 1; // scale factor for timing (to adjust for different CPU speeds)
int g_access_type = 0; // 0: read, 1: write
int g_cache_mode = 1; // 0: normal cached access, 1: 0 + clflushopt/clwb, 2: 0 + clflush, 3: non-temporal ld/st

// default values
size_t num_reads_outer = 10;
#if defined(__aarch64__)
size_t num_reads_inner = 10; // need amplication due to low timer resolution in ARM64
#else
size_t num_reads_inner = 1;
#endif
size_t mapping_size = (1ULL<<30); // 1GB default
size_t expected_sets = 16; // expected #sets in DRAM
size_t target_sets = 1; // target #sets to attack

int g_start_bit = 6; // search start bit
volatile int g_quit_signal = 0;
// ----------------------------------------------

#define MAX_HIST_SIZE 2000

std::vector <std::vector<pointer>> sets; // discovered sets

void *mapping = NULL; // 

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

uint64_t rdtsc() {
#if defined(__aarch64__)
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
    asm volatile("DC CVAC, %[ad]" : : [ad] "r" (p) : "memory");
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

// Non-temporal store (bypasses cache)
static inline void movnt_store(volatile void *p, uint64_t value) {
#if defined(__aarch64__)
    // ARM64: no native support, use DC instruction to bypass cache
    asm volatile("STR %[val], [%[ad]]\n\t"
                 "DC CVAC, %[ad]"
                 : : [ad] "r" (p), [val] "r" (value) : "memory");
#else
    // x86-64: use movnti for non-temporal store
    asm volatile("movnti %[val], (%[ad])"
                 : : [ad] "r" (p), [val] "r" (value) : "memory");
#endif
}

// Non-temporal load (bypasses cache) - x86 only
static inline uint64_t movnt_load(volatile void *p) {
#if defined(__aarch64__)
    // ARM64: no native support, use DC instruction to bypass cache
    uint64_t val;
    asm volatile("LDR %[val], [%[ad]]\n\t"
                 "DC CIVAC, %[ad]"
                 : [val] "=r" (val) : [ad] "r" (p) : "memory");
    return val;
#else
    // prefetchnta + cached load + clflushopt to simulate non-temporal load
    uint64_t val;
    asm volatile("prefetchnta (%0)" : : "r" (p) : "memory");
    val =  *(volatile uint64_t *)p;
    clflushopt(p);
    return val;
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

char *getAccessModeString(int mode) {
    if (g_cache_mode == 0) {
        switch (mode) {
        case 0:
            return (char *)"read";
        case 1:
            return (char *)"write";
        }
    } else if (g_cache_mode == 1) {
        switch (mode) {
        case 0:
            return (char *)"read+clflushopt";
        case 1:
            return (char *)"write+clwb";
        }
    } else if (g_cache_mode == 2) {
        switch (mode) {
        case 0:
            return (char *)"read+clflush";
        case 1:
            return (char *)"write+clflush";
        }
    } else if (g_cache_mode == 3) {
        switch (mode) {
        case 0:
            return (char *)"non-temporal read";
        case 1:
            return (char *)"non-temporal write";
        }
    }
    return (char *)"unknown";
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

    time_t t0 = utime();
    long dur_in_us = 0;
    // main loop
    int cur_access = g_access_type;
    while (!g_quit_signal) {
        if (cur_access != g_access_type) {
            dur_in_us = utime() - t0;
            long long t_bytes = (long long)local_set->size() * *ctr * 64LL;
            double mbps = (double)t_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
            printf("Thread %d: iters: %ld  Bandwidth: %.1f MB/s\n", cpu_id, *ctr, mbps);

            cur_access = g_access_type;
            logInfo("Thread %d: switching access type to %s\n",
                    cpu_id, (cur_access == 1) ? "write" : "read");
            t0 = utime();
            *ctr = 0; // reset counter
        }
        if (cur_access == 0) {
            // read attack
            for (size_t j = 0; j < n; ++j) {
                if (g_cache_mode == 0) {
                    // normal cached read
                    *((volatile int *)data[j]);
                } else if (g_cache_mode == 1) {
                    // read from the address and flush it
                    *((volatile int *)data[j]);
                    clflushopt((void *)data[j]);
                } else if (g_cache_mode == 2) {
                    // read from the address and flush it
                    *((volatile int *)data[j]);
                    clflush((void *)data[j]);
                } else if (g_cache_mode == 3) {
                    // non-temporal read
                    movnt_load((void *)data[j]);
                }
            }
        } else if (cur_access == 1) {
            // write attack
            for (size_t j = 0; j < n; ++j) {
                if (g_cache_mode == 0) {
                    // normal cached write
                    *((volatile int *)data[j]) = 0xdeadbeef;
                } else if (g_cache_mode == 1) {
                    // write to the address and clean it
                    *((volatile int *)data[j]) = 0xdeadbeef;
                    clwb((void *)data[j]); // if clwb is not supported, use clflushopt or clflush instead
                } else if (g_cache_mode == 2) {
                    // write to the address and flush it
                    *((volatile int *)data[j]) = 0xdeadbeef;
                    clflush((void *)data[j]);
                } else if (g_cache_mode == 3) {
                    // non-temporal write
                    movnt_store((void *)data[j], 0xdeadbeef);
                }
            }
        }
        (*ctr)++;
    }
    dur_in_us = utime() - t0;
    long long t_bytes = (long long)local_set->size() * (*ctr) * 64LL;
    double mbps = (double)t_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
    printf("Thread %d: iters: %ld  Bandwidth: %.1f MB/s\n", cpu_id, *ctr, mbps);

    return NULL;
}

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
    while ((c = getopt(argc, argv, "a:b:c:r:g:m:i:j:k:l:s:t:v:f:n:")) != EOF) {
        switch (c) {
        case 'a':
            g_access_type = (strncmp(optarg, "write", 5) == 0) ? 1 : 0;
            break;
        case 'b':
            g_start_bit = atoi(optarg);
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
        case 'l':
            target_sets = atoi(optarg);
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
            g_cache_mode = atoi(optarg);
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
                "Usage %s [-m <memory size in MB> | -g <memory size in GB>] [-i <number of outer loops>] [-j <number of inner loops>] [-k <target addresses per set>] [-s <expected sets>] [-t <threshold cycles>] [-f <cache mode>] [-l <target sets>]\n",
                argv[0]);
            exit(0);
            break;
        }
    }

    logInfo("Setting CPU affinity to core %d\n", cpu_affinity);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_affinity, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) != 0) {
        perror("sched_setaffinity");
        exit(1);
    }

    srand(time(NULL));
    g_page_size = sysconf(_SC_PAGESIZE);
    setupMapping();

    logInfo("Mapping has %zu MB\n", mapping_size / 1024 / 1024);

    logDebug("CPU: %s\n", getCPUModel());
    logDebug("Number of reads: %lu x %lu\n", num_reads_outer, num_reads_inner);
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

    // row hit timing
    t = getTiming(base, base + 64);
    logInfo("Average ROW hit cycles: %ld \n", t);

    int failed;
    uint64_t measure_count = 0;

    while (found_sets < target_sets) {
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
        } else if (samebank_threshold == -2) {
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
            if (t < found - 2) { // within 2 cycles of the threshold is okay
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

    int64_t total_addresses = 0;
    size_t min_set_size = SIZE_MAX;
    for (const auto& set : sets) {
        total_addresses += set.size();
        if (set.size() < min_set_size) {
            min_set_size = set.size();
        }
    }
    logInfo("Total %ld addresses found in %zu sets.\n", total_addresses, sets.size());
    printf("Accessing (%s) addresses in %ld sets (%ld addresses) with %d threads...\n",
            getAccessModeString(g_access_type),
            sets.size(), min_set_size * sets.size(), num_threads);

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArg> args(num_threads);
    std::vector<long> counters(num_threads, 0);
    std::vector<std::vector<pointer>> local_sets(num_threads);

    // divide addresses in sets into num_threads local sets
    for (int i = 0; i < num_threads; ++i) {
        local_sets[i].clear();
    }

    // distribute found addresses evenly to each thread's local set
    for (size_t j = 0; j < min_set_size; ++j) {
        // distribute addresses from each set to the local sets
        for(int i = 0; i < sets.size(); ++i ) {
            if (j == 0) logDebug("sets[%d][%zu] = 0x%lx\n", i, j, sets[i][j]);
            local_sets[j % num_threads].push_back(sets[i][j]);
        }
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

    // if SIGINT is received, set g_quit_signal to 1
    signal(SIGINT, [](int signum) {
        g_quit_signal = 1;
    });

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
        // double mbps = (double)t_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
        // printf("Thread %d: iters: %ld  Bandwidth: %.1f MB/s\n", i, counters[i], mbps);
        accessed_bytes += t_bytes;
    }
    // total b/w
    double total_mbps = (double)accessed_bytes / (double)dur_in_us * 1000000.0 / (1024.0*1024.0);
    printf("Total aggregate bandwidth: %.1f MB/s\n", total_mbps);
    return 0;
}
