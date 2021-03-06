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


#if defined(__aarch64__)
#define USE_MEDIAN         1
#endif
#define USE_KAM            0    // require to load kam.ko (github.com/heechul/bank_test)
#define DISPLAY_PROGRESS   1    // reamining time update.
#define MAX_OUTER_LOOP     1000
#define POINTER_SIZE       (sizeof(void*) * 8) // #of bits of a pointer
#define MAX_XOR_BITS       7    // orig: 7

// ------------ global settings ----------------
int verbosity = 4;

// default values
size_t num_reads_outer = 10;
size_t num_reads_inner = 10;
size_t mapping_size = (1<<23); // 8MB default
size_t expected_sets = 16;
int g_start_bit = 11; // search start bit
int g_use_linear_addr = 1;
int g_display_progress = 1;
int g_use_kam = 0;

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
    while (getline(&buffer, &n, f)) {
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
    if (g_use_kam) {
        int fd = open("/dev/kam", O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "Couldn't open device file\n");
            exit(1);
        }

        mapping = mmap(NULL, mapping_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);

        /* We don't close the file. We let it close when exit */
        if (mapping == MAP_FAILED) {
            perror("Couldn't allocate memory from device\n");
            exit(2);
        }

        // Get the physcal address of start address
        int iret = ioctl(fd, 0, &phy_start_addr);
        if (iret < 0) {
            perror("Couldn't find the physical address of start\n");
            exit(3);
        }
    
        // *(int *)ret = 0x12345678;
        logDebug("Value [%p]=%x\n", mapping, *(int *)mapping);
    
        logDebug("%s", "Initialize large memory block...\n");
        for (size_t index = 0; index < mapping_size; index += 0x1000) {
            pointer *temporary =
                reinterpret_cast<pointer *>(static_cast<uint8_t *>(mapping)
                                            + index);
            temporary[0] = index;
        }
        logDebug("%s", " done!\n");
    } else {
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
                    logInfo("%s small page mapping\n", "4KB");
            } else
                logInfo("%s huge page mapping\n", "2MB");
        } else
            logInfo("%s huge page mapping\n", "1GB");

        assert(mapping != (void *) -1);

        logDebug("%s", "Initialize large memory block...\n");
        for (size_t index = 0; index < mapping_size; index += 0x1000) {
            pointer *temporary =
                reinterpret_cast<pointer *>(static_cast<uint8_t *>(mapping)
                                            + index);
            temporary[0] = index;
        }
        logDebug("%s", " done!\n");
    }        
}

// ----------------------------------------------
size_t frameNumberFromPagemap(size_t value) {
    return value & ((1ULL << 54) - 1);
}

pointer getPhysicalAddr(pointer virtual_addr) {
    if (g_use_kam) {
        return (virtual_addr - (unsigned long int)mapping) + phy_start_addr;
    } else {
        pointer value;
        off_t offset = (virtual_addr / 4096) * sizeof(value);
        int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
        assert(got == 8);

        // Check the "page present" flag.
        assert(value & (1ULL << 63));

        pointer frame_num = frameNumberFromPagemap(value);
        return (frame_num * 4096) | (virtual_addr & (4095));
    }
}

// ----------------------------------------------
void initPagemap() {
    g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    assert(g_pagemap_fd >= 0);
}

// ----------------------------------------------
long utime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
}


// ----------------------------------------------

#if defined(__aarch64__)
#define USE_FAST_COUNTER   1

static volatile uint64_t counter = 0;
static pthread_t count_thread;

static void *countthread(void *dummy) {
    uint64_t local_counter = 0;
    while (1) {
#if USE_FAST_COUNTER==1
        local_counter++;
        counter = local_counter;
#else
        counter++;
#endif
    }
    return NULL;
}
#endif

uint64_t rdtsc() {
#if defined(__aarch64__)
    asm volatile ("DSB SY");	
    return counter;
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


size_t low_thresh = 0, high_thresh = WINT_MAX;

static int comparator(const void *p, const void *q)
{
  return *(int *)p > *(int *)q;
}

// ----------------------------------------------
uint64_t getTiming(pointer first, pointer second) {
    size_t min_res = (-1ull);
    size_t ticks[MAX_OUTER_LOOP];
    assert(num_reads_outer <= MAX_OUTER_LOOP);

    for (int i = 0; i < num_reads_outer; i++) {
        size_t number_of_reads = num_reads_inner;
        volatile size_t *f = (volatile size_t *) first;
        volatile size_t *s = (volatile size_t *) second;

        sched_yield();
        
        size_t t0 = rdtsc();

        while (number_of_reads-- > 0) {
            *f;
            *(f + number_of_reads);

            *s;
            *(s + number_of_reads);

#if defined(__aarch64__)
            asm volatile("DC CIVAC, %[ad]" : : [ad] "r" (f) : "memory");
            asm volatile("DC CIVAC, %[ad]" : : [ad] "r" (s) : "memory");
#else
            asm volatile("clflush (%0)" : : "r" (f) : "memory");
            asm volatile("clflush (%0)" : : "r" (s) : "memory");
            asm volatile("mfence");
#endif
        }

        uint64_t res = (rdtsc2() - t0) / (num_reads_inner);

#if USE_MEDIAN==1
        if (res <= low_thresh || res > high_thresh) {
            i--;	
            continue;
        }
        ticks[i] = res;
#else        
        if (res < min_res)
            min_res = res;
#endif
    }

#if USE_MEDIAN==1
    qsort((void *)ticks, num_reads_outer, sizeof(ticks[0]), comparator);
    min_res = ticks[num_reads_outer/2];
#endif
    
    return min_res;
}

// ----------------------------------------------
void getRandomAddress(pointer *virt, pointer *phys) {
    size_t offset = (size_t)(rand() % (mapping_size / 128)) * 128;
    *virt = (pointer) mapping + offset;
    *phys = getPhysicalAddr(*virt);
}

// ----------------------------------------------
void clearLine() {
    printf("\033[2K\r");
}

// ----------------------------------------------
char *formatTime(long ms) {
    static char buffer[64];
    long minutes = ms / 60000;
    if (minutes == 0) {
        sprintf(buffer, "%.1fs", ms / 1000.0);
    } else {
        sprintf(buffer, "%lum %.1lfs", minutes,
                (ms - minutes * 60000) / 1000.0);
    }
    return buffer;
}


// ----------------------------------------------
pointer next_set_of_n_elements(pointer x) {
    pointer smallest, ripple, new_smallest, ones;

    if (x == 0)
        return 0;
    smallest = (x & -x);
    ripple = x + smallest;
    new_smallest = (ripple & -ripple);
    ones = ((new_smallest / smallest) >> 1) - 1;
    return ripple | ones;
}

// ----------------------------------------------
int pop(unsigned x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x0000003F;
}

// ----------------------------------------------
int xor64(pointer addr) {
    return (pop(addr & 0xffffffff) + pop((addr >> 32) & 0xffffffff)) & 1;
}

// ----------------------------------------------
int apply_bitmask(pointer addr, pointer mask) {
    return xor64(addr & mask);
}

// for debugging: only valid on raspberry pi 4
// int get_bank_num(pointer addr) {
//     int bank =
//         ((addr >> 11) & 0x1) |
//         (((addr >> 12) & 0x1) << 1) |
//         (((addr >> 13) & 0x1) << 2) |
//         (((addr >> 14) & 0x1) << 3); 
//     return bank;
// }

// ----------------------------------------------
char *name_bits(pointer mask) {
    static char name[256], bn[8];
    strcpy(name, "");
    for (int i = 0; i < sizeof(pointer) * 8; i++) {
        if (mask & (1ull << i)) {
            sprintf(bn, "%d ", i);
            strcat(name, bn);
        }
    }
    return name;
}

// ----------------------------------------------
std::vector <pointer> find_function(int bits, int pointer_bit, int align_bit) {
    // try to find a 2 bit function
    pointer start_mask = (1 << bits) - 1;
    std::set <pointer> func_pool;
    for (int set = 0; set < sets.size(); set++) {
        std::set <pointer> set_func;
        unsigned int mask = start_mask;

        // logDebug("Set %d: 0x%lx count: %ld\n", set + 1, sets[set][0], sets[set].size());
        while (1) {
            if (sets[set].size() == 0) break;
            // check if mask produces same result for all addresses in set
            int ref = apply_bitmask(sets[set][0] >> align_bit, mask);

            bool insert = true;
            for (int a = 1; a < sets[set].size(); a++) {
                if (apply_bitmask(sets[set][a] >> align_bit, mask) != ref) {
                    insert = false;
                    break;
                }
            }
            if (insert) {
                set_func.insert(mask);
            }

            mask = next_set_of_n_elements(mask);
            if (mask <= start_mask
                || (mask & (1 << (pointer_bit - 1 - align_bit)))) {
                break;
            }
        }
        // intersect with function pool
        if (func_pool.empty()) {
            func_pool.insert(set_func.begin(), set_func.end());
        }
        std::set_intersection(set_func.begin(), set_func.end(),
                              func_pool.begin(), func_pool.end(),
                              std::inserter(func_pool, func_pool.begin()));
    }
    std::vector <pointer> func;
    for (std::set<pointer>::iterator f = func_pool.begin();
         f != func_pool.end(); f++) {
        func.push_back((*f) << g_start_bit);
    }
    return func;
}

// ----------------------------------------------
std::vector<double> prob_function(std::vector <pointer> masks, int align_bit) {
    std::vector<double> prob;
    for (std::vector<pointer>::iterator it = masks.begin(); it != masks.end();
         it++) {
        pointer mask = *it;
        int count = 0;
        for (int set = 0; set < sets.size(); set++) {
            if (sets[set].size() == 0) continue;
            if (apply_bitmask(sets[set][0], mask)) // << BUG FIX
                count++;
        }
        // logDebug("%s: %.2f\n", name_bits(mask), (double) count / sets.size());
        prob.push_back((double) count / sets.size());
    }
    return prob;
}

// ----------------------------------------------
int main(int argc, char *argv[]) {
    size_t tries, t;
    std::set <addrpair> addr_pool;
    std::map<int, std::list<addrpair> > timing;
    size_t hist[MAX_HIST_SIZE];
    int c;
    int samebank_threshold = -1;

    while ((c = getopt(argc, argv, "b:d:m:i:j:ks:t:rv:")) != EOF) {
        switch (c) {
        case 'b':
            g_start_bit = atof(optarg);
            break;
        case 'd':
            g_display_progress = atoi(optarg);
            break;
        case 'm':
            mapping_size = atol(optarg) * 1024 * 1024;
            break;
        case 'i':
            num_reads_outer = atol(optarg);
            break;
        case 'j':
            num_reads_inner = atol(optarg);
            break;
        case 'k':
            g_use_kam = 1;
            break;
        case 's':
            expected_sets = atoi(optarg);
            break;
        case 't':
            samebank_threshold = atoi(optarg);
            break;
        case 'r':
            g_use_linear_addr = 0;
            break;
        case 'v':
            verbosity = atoi(optarg);
            break;
        case ':':
            printf("Missing option.\n");
            exit(1);
            break;
        default:
            printf(
                "Usage %s [-m <memory size in MB>] [-i <number of outer loops>] [-j <number of inner loops>] [-s <expected sets>] [-t <threshold cycles> [-k] [-r]]\n",
                argv[0]);
            printf("-k : use KAM module\n");
            printf("-r : use random addresses\n");
            exit(0);
            break;
        }
    }

    logDebug("CPU: %s\n", getCPUModel());
    logDebug("Number of reads: %lu x %lu\n", num_reads_outer, num_reads_inner)
    logDebug("Expected sets: %lu\n", expected_sets);

    srand(time(NULL));
    initPagemap();
    setupMapping();

    logInfo("Mapping has %zu MB\n", mapping_size / 1024 / 1024);
    
    pointer first, second;
    pointer first_phys, second_phys;
    pointer base, base_phys;

    size_t remaining_tries;

    long times[ETA_BUFFER], time_start;
    int time_ptr = 0;
    int time_valid = 0;

    int found_sets = 0;
    int found_siblings = 0;

    logInfo("Address pool method: %s\n", (g_use_linear_addr)?"linear":"random");
    if (g_use_linear_addr) {
        tries = mapping_size / (1<<g_start_bit);

        base = (pointer)mapping;
        base_phys = getPhysicalAddr(base);
        
        while (addr_pool.size() < tries) {
            int idx = addr_pool.size();
            second = base + idx * (1<<g_start_bit);
            second_phys = getPhysicalAddr(second); 
#if 0
            logDebug("addr_pool[%ld]: 0x%0lx Bank %d\n",
                     addr_pool.size(), second_phys, get_bank_num(second_phys));
#endif
            addr_pool.insert(std::make_pair(second, second_phys));
        }
    } else {
        tries = expected_sets * 125; // DEBUG: original 125.
    
        // build address pool
        while (int cur_count = addr_pool.size() < tries) {
            getRandomAddress(&second, &second_phys);
            addr_pool.insert(std::make_pair(second, second_phys));
            // if (cur_count != addr_pool.size())
            //     logDebug("addr_pool[%ld]: 0x%0lx Bank %d\n",
            //              addr_pool.size(), second_phys, get_bank_num(second_phys));
        }

        auto ait = addr_pool.begin();
        // std::advance(ait, rand() % addr_pool.size());
        base = ait->first;
        base_phys = ait->second;
    }

    logDebug("Address pool size: %lu\n", addr_pool.size());

    setpriority(PRIO_PROCESS, 0, -20);

#if defined(__aarch64__)
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


    t = getTiming(base, base + sizeof(size_t));
    low_thresh = t * 0.5;
    high_thresh = t * 3;
    logInfo("Average cycles: %ld  low_threshold: %ld high_threshold: %ld\n",
            t, low_thresh, high_thresh);
    
    int failed;
    uint64_t sum_ticks = 0, measure_count = 0;

    while (found_sets < expected_sets) {
        for (size_t i = 0; i < MAX_HIST_SIZE; ++i)
            hist[i] = 0;
        failed = 0;
    search_set:
        failed++;
        if (failed > 10) {
            logWarning("%s\n", "Couldn't find set after 10 tries, giving up, sorry!");
            break;
        }
        logInfo("Searching for set %d (try %d): base_phy=0x%lx\n",
                found_sets + 1, failed, base_phys);
        timing.clear();
        remaining_tries = addr_pool.size(); // tries;

        // measure access times
        sched_yield();
        std::set <addrpair> used_addr;
        used_addr.clear();
        while (--remaining_tries) {
            sched_yield();
            time_start = utime();

            // get random address from address pool (prevents any prefetch or something)
            auto pool_front = addr_pool.begin();
            if (0) // g_use_linear_addr)
                std::advance(pool_front, (addr_pool.size()-remaining_tries));
            else
                std::advance(pool_front, rand() % addr_pool.size());

            first = pool_front->first;
            first_phys = pool_front->second;

            // measure timing
            // sched_yield();
            t = getTiming(base, first);
            sum_ticks += t;
            measure_count++;

            // sched_yield();
            timing[t].push_back(std::make_pair(base_phys, first_phys));

            times[time_ptr] = utime() - time_start;
            time_ptr++;
            if (time_ptr == ETA_BUFFER) {
                time_ptr = 0;
                time_valid = 1;
            }

            // sched_yield();
            if (g_display_progress) {
                clearLine();
                if (time_valid) {
                    long mean = 0;
                    for (int i = 0; i < ETA_BUFFER; i++) {
                        mean += times[i];
                    }
                    mean /= ETA_BUFFER;
                    printf("%lu%% (ETA: %s)",
                           (tries - remaining_tries + 1) * 100 / tries,
                           formatTime(mean * remaining_tries));
                } else {
                    printf("%lu%% (ETA: %c)",
                           (tries - remaining_tries + 1) * 100 / tries,
                           "|/-\\"[time_ptr % 4]);
                }
                fflush(stdout);
            }
        }
        printf("\n");

        // identify sets -> must be on the right, separated in the histogram
        std::vector <pointer> new_set;
        std::map < int, std::list < addrpair > > ::iterator hit;
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

        // scale histogram
        double scale_v = (double) (100.0)
                         / (max_v > 0 ? (double) max_v : 100.0);
        assert(scale_v >= 0);
        while (hist[++min] <= 1);
        while (hist[--max] <= 1);

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
#if 0
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

            if (!found) {
                logWarning("%s\n", "No set found, trying again...");
                goto search_set;
            }
#else
            found = (sum_ticks / measure_count) * 135 / 100;
#endif
        }

        new_set.push_back(base_phys); // this is needed. another bug in the original code

        // remove found addresses from pool
        for (hit = timing.begin(); hit != timing.end(); hit++) {
            if (hit->first >= found && hit->first <= max) {
                for (std::list<addrpair>::iterator it = hit->second.begin();
                     it != hit->second.end(); it++) {
                    new_set.push_back(it->second);
                }
            }
        }

	logInfo("found(cycles): %d newset_sz: %lu (expected_sz: %lu) pool_sz: %lu\n",
                 found, new_set.size(), tries/expected_sets, addr_pool.size());

        if (new_set.size() <= tries / expected_sets * 0.8) {
            logWarning("Set must be wrong, contains too few addresses (%lu). Try again...\n", new_set.size());
            goto search_set;
        }
        if (new_set.size() > tries / expected_sets * 1.2) {
            /* addr_pool.size() / expected..*/
	    logWarning("Set must be wrong, contains too many addresses (expected: %lu/found: %ld). Try again...\n", tries / expected_sets, new_set.size());

            goto search_set;
        }


        for (auto it = addr_pool.begin(); it != addr_pool.end();) {
            int erased = 0;
            for (auto nit = new_set.begin(); nit != new_set.end(); nit++) {
                if (*nit == it->second) {
                    it = addr_pool.erase(it);
                    erased = 1;
                    break;
                }
            }
            if (!erased) {
                it++;
            }
        }

        // save identified set if one was found
        sets.push_back(new_set);
        found_siblings += new_set.size();
        
        // choose base address from remaining addresses
        auto ait = addr_pool.begin();
        // std::advance(ait, rand() % addr_pool.size());
        base = ait->first;
        base_phys = ait->second;

        found_sets++;
    }
    logDebug("Done measuring. found_sets: %d found_siblings: %d\n",
             found_sets, found_siblings);

    for (int set = 0; set < sets.size(); set++) {
        logInfo("Set %d: 0x%lx count: %ld\n",
                set + 1, sets[set][0], sets[set].size());
        // for (int j = 0; j < sets[set].size(); j++) {
        //     logDebug("  0x%lx Bank %d\n",
        //              sets[set][j], get_bank_num(sets[set][j]));
        // }
    }
    
    // try to find a xor function
    std::map<int, std::vector<double> > prob;
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        functions[bits] = find_function(bits, POINTER_SIZE, g_start_bit);
        prob[bits] = prob_function(functions[bits], g_start_bit);
    }

    // filter out false positives
    std::vector <pointer> false_positives;
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        for (int j = 0; j < prob[bits].size(); j++) {
            if (prob[bits][j] <= 0.01 || prob[bits][j] >= 0.99) {
                // false positives, this bits are always 0 or 1
                false_positives.push_back(functions[bits][j]);
                // logDebug("False positive function: %s\n", name_bits(functions[bits][j]));
            }
        }
    }

    // optimize functions -> remove all, that are combinations of others
    std::map<int, std::vector<pointer> > duplicates;

    // get dimensions
    uint64_t rows = 0, cols = 0;
    // find number of functions, highest bit and lowest bit
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        rows += functions[bits].size();
	logDebug("functions[%d].size(): %ld\n", bits, functions[bits].size());
        for (pointer f : functions[bits]) {
            if (f > cols) cols = f;
        }
    }
    cols = (int) (std::log2(cols) + 0.5) + 1;

    // allocate matrix
    int **matrix = new int *[rows];
    for (int r = 0; r < rows; r++) {
        matrix[r] = new int[cols];
    }

    // build matrix
    int r = 0;
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        for (pointer f : functions[bits]) {

            for (int b = 0; b < cols; b++) {
                matrix[r][cols - b - 1] = (f & (1ul << b)) ? 1 : 0;
            }
            r++;
        }
    }

    // transpose matrix
    int **matrix_t = new int *[cols];
    for (int r = 0; r < cols; r++) {
        matrix_t[r] = new int[rows];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_t[j][i] = matrix[i][j];
        }
    }

    int i = 0;
    int j = 0;
    int t_rows = cols;
    int t_cols = rows;
    std::vector<int> jb;
    // gauss-jordan
    while (i < t_rows && j < t_cols) {

        // get value and index of largest element in current column j
        int max_v = -9999, max_p = 0;
        for (int p = i; p < t_rows; p++) {
            if (matrix_t[p][j] > max_v) {
                max_v = matrix_t[p][j];
                max_p = p;
            }
        }
        if (max_v == 0) {
            // column is zero, goto next
            j++;
        } else {
            // remember column index
            jb.push_back(j);
            // swap i-th and max_p-th row
            int *temp_row = matrix_t[i];
            matrix_t[i] = matrix_t[max_p];
            matrix_t[max_p] = temp_row;

            // subtract multiples of the pivot row from all other rows
            for (int k = 0; k < t_rows; k++) {
                if (k == i) continue;
                int kj = matrix_t[k][j];
                for (int p = j; p < t_cols; p++) {
                    matrix_t[k][p] ^= kj & matrix_t[i][p];
                }
            }
            i++;
            j++;
        }
    }

    std::cout << "reduced to " << jb.size() << " functions" << std::endl;

    // remove duplicates
    r = 0;
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        for (pointer f : functions[bits]) {
            if (std::find(jb.begin(), jb.end(), r) == jb.end()) {
                duplicates[bits].push_back(f);
            }
            r++;
        }
    }

    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;


    // display found functions
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
	printf("Bits: %d, sz=%d\n", bits, (int)functions[bits].size());
	
        for (int i = 0; i < functions[bits].size(); i++) {
            bool show = true;
            for (int fp = 0; fp < false_positives.size(); fp++) {
                if ((functions[bits][i] & false_positives[fp]) == false_positives[fp]) {
                    show = false;
                    break;
                }
            }
            if (!show) continue;

            for (int dup = 0; dup < duplicates[bits].size(); dup++) {
                if (functions[bits][i] == duplicates[bits][dup]) {
                    show = false;
                };
            }
            if (!show)
                continue;

            printf("%s (Correct: %d%%)\n", name_bits(functions[bits][i]),
                   (int) (100 - fabs(0.5 - prob[bits][i]) * 200));
        }
    }

    
    fprintf(stderr, "Finishing\n");
    exit(1);
    return 0;

}
