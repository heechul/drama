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
#include <queue>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>
#include <sstream>
#include <iterator>
#include <math.h>
#include <signal.h>
#include <sys/ioctl.h>
#include "measure.h"


#define POINTER_SIZE       (sizeof(void*) * 8) // #of bits of a pointer
#define MAX_XOR_BITS       15    // orig: 7
// ------------ global settings ----------------
int verbosity = 1;
size_t g_page_size;

int g_scale_factor = 1;

int enable_bg_detection = 1;
int bg_samples_per_pair = 5;

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

// ----------------------------------------------

#define MAX_HIST_SIZE 2000

std::vector <std::vector<addrpair>> sets;
std::map<int, std::vector<pointer> > functions;

struct PairTiming {
    int set_a;
    int set_b;
    double timing;
};

struct BGThresholdInfo {
    double threshold;
    double low_center;
    double high_center;
    bool valid;
};

struct BGFunctionCandidate {
    pointer mask;
    int bits;
    double accuracy;
    double coverage;
};

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
size_t frameNumberFromPagemap(size_t value) {
    return value & ((1ULL << 54) - 1);
}

pointer getPhysicalAddr(pointer virtual_addr) {
    pointer value;
    off_t offset = (virtual_addr / g_page_size) * sizeof(value);
    int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
    assert(got == 8);

    // Check the "page present" flag.
    assert(value & (1ULL << 63));

    pointer frame_num = frameNumberFromPagemap(value);
    return (frame_num * g_page_size) | (virtual_addr & (g_page_size - 1));
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
void getRandomAddress(pointer *virt, pointer *phys) {
    size_t unit_size = (1ULL << g_start_bit);
    size_t offset = (size_t)(rand() % (mapping_size / unit_size)) * unit_size;
    *virt = (pointer) mapping + offset;
    *phys = getPhysicalAddr(*virt);
}

// Count bits in 64-bit word
static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

// ----------------------------------------------
int xor64(pointer addr) {
    return (popcount64(addr)) & 1;
}

// xor64: compute parity (XOR of all bits) of a 64-bit value.
// It uses `pop` to count ones in the low and high 32-bit halves and
// returns parity (0 or 1). This is used to evaluate parity-based
// bank-indexing functions.

// ----------------------------------------------
int apply_bitmask(pointer addr, pointer mask) {
    return xor64(addr & mask);
}

// apply_bitmask: apply `mask` to `addr` and return the parity (0/1)
// of the selected bits. In the context of bank-function discovery, the
// code checks whether this parity is constant for all addresses in a set.

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

// name_bits: return a human-readable string listing the bit positions
// set in `mask`. Example: mask with bits 5 and 7 => "5 7". Used for
// printing and saving discovered functions.


double min_pair_timing(const std::vector<addrpair>& first_set,
                          const std::vector<addrpair>& second_set,
                          int samples_per_pair) {
    if (first_set.empty() || second_set.empty()) return 0.0;
    if (samples_per_pair <= 0) samples_per_pair = 1;

    double min_timing = -1.0;
    for (int s = 0; s < samples_per_pair; s++) {
        const addrpair& lhs = first_set[rand() % first_set.size()];
        const addrpair& rhs = second_set[rand() % second_set.size()];
        double timing = static_cast<double>(getTiming(lhs.first, rhs.first));
        if (min_timing < 0 || timing < min_timing) {
            min_timing = timing;
        }
    }

    return min_timing;
}

std::vector<PairTiming> collect_pair_timings(
        const std::vector<std::vector<addrpair>>& input_sets,
        int samples_per_pair) {
    std::vector<PairTiming> measurements;
    if (input_sets.size() < 2) return measurements;

    for (size_t i = 0; i < input_sets.size(); i++) {
        for (size_t j = i + 1; j < input_sets.size(); j++) {
            double timing = min_pair_timing(input_sets[i], input_sets[j], samples_per_pair);
            if (timing <= 0.0) continue;
            measurements.push_back({static_cast<int>(i), static_cast<int>(j), timing});
            logInfo("BG timing set %zu vs %zu => %.2f\n", i + 1, j + 1, timing);
        }
    }

    logInfo("Collected %zu inter-set timing samples for BG detection (samples=%d).\n",
            measurements.size(), samples_per_pair);
    return measurements;
}

BGThresholdInfo compute_bg_threshold(const std::vector<PairTiming>& pair_timings) {
    BGThresholdInfo info{0.0, 0.0, 0.0, false};
    if (pair_timings.size() < 2) return info;

    double min_v = pair_timings[0].timing;
    double max_v = pair_timings[0].timing;
    for (const auto& pt : pair_timings) {
        if (pt.timing < min_v) min_v = pt.timing;
        if (pt.timing > max_v) max_v = pt.timing;
    }

    if (fabs(max_v - min_v) < 1.0) return info;

    double c1 = min_v;
    double c2 = max_v;
    const double EPS = 1e-3;

    for (int iter = 0; iter < 100; iter++) {
        double sum1 = 0.0, sum2 = 0.0;
        int cnt1 = 0, cnt2 = 0;
        for (const auto& pt : pair_timings) {
            double d1 = fabs(pt.timing - c1);
            double d2 = fabs(pt.timing - c2);
            if (d1 <= d2) {
                sum1 += pt.timing;
                cnt1++;
            } else {
                sum2 += pt.timing;
                cnt2++;
            }
        }
        if (cnt1 == 0 || cnt2 == 0) {
            return info;
        }
        double new_c1 = sum1 / cnt1;
        double new_c2 = sum2 / cnt2;
        if (fabs(new_c1 - c1) < EPS && fabs(new_c2 - c2) < EPS) break;
        c1 = new_c1;
        c2 = new_c2;
    }

    if (c1 > c2) std::swap(c1, c2);
    if (fabs(c2 - c1) < 1.0) return info;

    info.low_center = c1;
    info.high_center = c2;
    info.threshold = c1 + (c2 - c1) / 2.0 + 4.0; // margin
    info.valid = true;
    return info;
}

std::vector<int> derive_bg_labels(const std::vector<PairTiming>& pair_timings,
                                  const BGThresholdInfo& info,
                                  size_t set_count) {
    std::vector<int> labels(set_count, -1);
    if (!info.valid || set_count == 0) return labels;

    std::vector<std::vector<bool>> adjacency(set_count, std::vector<bool>(set_count, false));
    for (const auto& pt : pair_timings) {
        if (pt.set_a < 0 || pt.set_b < 0) continue;
        if (pt.set_a >= static_cast<int>(set_count) ||
            pt.set_b >= static_cast<int>(set_count)) continue;
        bool same_group = pt.timing >= info.threshold;
        adjacency[pt.set_a][pt.set_b] = same_group;
        adjacency[pt.set_b][pt.set_a] = same_group;
    }

    int group_id = 0;
    std::queue<int> bfs;
    for (size_t i = 0; i < set_count; i++) {
        if (labels[i] != -1) continue;
        labels[i] = group_id;
        bfs.push(static_cast<int>(i));
        while (!bfs.empty()) {
            int cur = bfs.front();
            bfs.pop();
            for (size_t j = 0; j < set_count; j++) {
                if (!adjacency[cur][j]) continue;
                if (labels[j] != -1) continue;
                labels[j] = group_id;
                bfs.push(static_cast<int>(j));
            }
        }
        group_id++;
    }

    logInfo("Identified %d bank groups from timing adjacency.\n", group_id);
    return labels;
}

int compute_set_parity(const std::vector<addrpair>& set_entries, pointer mask) {
    if (set_entries.empty()) return 0;
    int ones = 0;
    for (const auto& entry : set_entries) {
        if (apply_bitmask(entry.second, mask)) ones++;
    }
    return (ones * 2 >= static_cast<int>(set_entries.size())) ? 1 : 0;
}

std::vector<BGFunctionCandidate> rank_bg_functions(
        const std::vector<int>& bg_labels,
        const std::vector<std::vector<addrpair>>& input_sets,
        const std::map<int, std::vector<pointer>>& functions,
        const std::vector<pointer>& false_positives,
        const std::map<int, std::vector<pointer>>& duplicates) {
    std::vector<BGFunctionCandidate> ranked;
    if (bg_labels.empty()) return ranked;

    std::set<pointer> false_positive_set(false_positives.begin(), false_positives.end());
    std::set<pointer> duplicate_set;
    for (const auto& kv : duplicates) {
        for (pointer mask : kv.second) {
            duplicate_set.insert(mask);
        }
    }

    size_t labeled_sets = 0;
    std::set<int> unique_groups;
    for (size_t i = 0; i < bg_labels.size(); i++) {
        if (bg_labels[i] >= 0 && i < input_sets.size() && !input_sets[i].empty()) {
            labeled_sets++;
            unique_groups.insert(bg_labels[i]);
        }
    }

    if (labeled_sets < 2 || unique_groups.size() < 2) return ranked;

    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        auto fn_it = functions.find(bits);
        if (fn_it == functions.end()) continue;
        for (pointer mask : fn_it->second) {
            if (false_positive_set.count(mask) > 0) continue;
            if (duplicate_set.count(mask) > 0) continue;

            std::vector<int> parity_ready(input_sets.size(), 0);
            std::vector<int> parity_value(input_sets.size(), 0);
            size_t participating_sets = 0;

            for (size_t set_idx = 0; set_idx < input_sets.size(); set_idx++) {
                if (set_idx >= bg_labels.size()) break;
                if (bg_labels[set_idx] < 0) continue;
                if (input_sets[set_idx].empty()) continue;
                parity_value[set_idx] = compute_set_parity(input_sets[set_idx], mask);
                parity_ready[set_idx] = 1;
                participating_sets++;
            }

            if (participating_sets < 2) continue;

            std::set<int> parity_values;
            for (size_t idx = 0; idx < parity_ready.size(); idx++) {
                if (parity_ready[idx]) parity_values.insert(parity_value[idx]);
            }
            if (parity_values.size() < 2) continue;

            int matches = 0;
            int comparisons = 0;
            for (size_t i = 0; i < input_sets.size(); i++) {
                if (!parity_ready[i]) continue;
                for (size_t j = i + 1; j < input_sets.size(); j++) {
                    if (!parity_ready[j]) continue;
                    bool same_group = (bg_labels[i] == bg_labels[j]);
                    bool same_parity = (parity_value[i] == parity_value[j]);
                    if (same_group == same_parity) matches++;
                    comparisons++;
                }
            }

            if (comparisons == 0) continue;
            double accuracy = static_cast<double>(matches) / static_cast<double>(comparisons);
            double coverage = static_cast<double>(participating_sets) /
                               static_cast<double>(labeled_sets);

            if (accuracy >= 0.50) {
                ranked.push_back({mask, bits, accuracy, coverage});
            }
        }
    }

    std::sort(ranked.begin(), ranked.end(), [](const BGFunctionCandidate& lhs,
                                               const BGFunctionCandidate& rhs) {
        if (fabs(lhs.accuracy - rhs.accuracy) > 1e-6)
            return lhs.accuracy > rhs.accuracy;
        if (fabs(lhs.coverage - rhs.coverage) > 1e-6)
            return lhs.coverage > rhs.coverage;
        return lhs.bits < rhs.bits;
    });

    return ranked;
}

void print_bg_candidates(const std::vector<BGFunctionCandidate>& candidates) {
    if (candidates.empty()) {
        printf("No bank-group candidate functions matched the timing evidence.\n");
        return;
    }

    printf("\n=== Bank Group Candidate Functions ===\n");
    for (const auto& cand : candidates) {
        printf("%s (bits=%d, accuracy=%.1f%%, coverage=%.1f%%)\n",
               name_bits(cand.mask),
               cand.bits,
               cand.accuracy * 100.0,
               cand.coverage * 100.0);
    }
}

void save_bank_group_functions(const char* base_filename,
                               const std::vector<BGFunctionCandidate>& candidates) {
    if (candidates.empty()) return;

    std::string filename = (base_filename && strlen(base_filename) > 0)
                               ? std::string(base_filename)
                               : std::string("map.txt");
    filename += ".bg";

    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing bank-group functions\n",
                filename.c_str());
        return;
    }

    for (const auto& cand : candidates) {
        std::string bits = name_bits(cand.mask);
        if (!bits.empty() && bits.back() == ' ') bits.pop_back();
        fprintf(fp, "%s # bits=%d accuracy=%.4f coverage=%.4f\n",
                bits.c_str(), cand.bits, cand.accuracy, cand.coverage);
    }
    fclose(fp);
    printf("Bank-group mapping functions saved to %s\n", filename.c_str());
}


// ----------------------------------------------
std::vector<double> prob_function(std::vector <pointer> masks, int align_bit) {
    std::vector<double> prob;
    for (std::vector<pointer>::iterator it = masks.begin(); it != masks.end();
         it++) {
        pointer mask = *it;
        int count = 0;            // number of sets with parity == 1
        int used_sets = 0;        // number of non-empty sets considered

        for (int set = 0; set < sets.size(); set++) {
            if (sets[set].size() == 0) continue;

            // majority voting across addresses in this set for robustness
            int ones = 0;
            for (size_t a = 0; a < sets[set].size(); a++) {
                if (apply_bitmask(sets[set][a].second, mask)) ones++;
            }
#if 0
            // if majority of addresses in the set have parity 1, count this set as 1
            int parity = (ones * 2 >= (int)sets[set].size()) ? 1 : 0;
#else
            // all addresses in the set must have parity 1 to count this set as 1
            int parity = (ones == (int)sets[set].size()) ? 1 : 0;
#endif
            if (parity) count++;
            used_sets++;
        }

        double p = 0.0;
        if (used_sets > 0) p = (double) count / (double) used_sets;
        // logDebug("%s: %.2f (used_sets=%d)\n", name_bits(mask), p, used_sets);
        prob.push_back(p);
    }
    return prob;
}

// ---------------- GF(2) linear solver helpers ----------------

// Build GF(2) matrix rows as uint64_t if B <= 64. We restrict to B<=64
// in the simple implementation below to pack columns into a single word.
// For larger B the code would need vector<uint64_t> rows.

// find_function_linear:
// Build linear equations v dot x = 0 (mod2) from set address pairs
// and solve for x (unknown mask bits) using Gaussian elimination over GF(2).
// Then enumerate nullspace combinations (when small) to find solutions with
// Hamming weight == bits. Returns masks shifted left by g_start_bit.
std::vector<pointer> find_function_linear(int bits, int pointer_bit, int align_bit) {
    std::vector<pointer> results;

    // remaining bit window: consider bits from align_bit .. g_end_bit inclusive
    int B = (g_end_bit + 1 - align_bit);
    if (B <= 0 || B > 64) {
        // fallback: empty (unsupported B for this simple solver)
        logWarning("find_function_linear: unsupported bit-window B=%d\n", B);
        return results;
    }

    // Build equation matrix: for each set, for each address a != base, add equation
    // ( (base ^ a) >> align_bit ) & ((1<<B)-1)  dot x == 0
    std::vector<uint64_t> rows;
    uint64_t mask_window = (B == 64) ? ~0ULL : ((1ULL << B) - 1);

    for (size_t s = 0; s < sets.size(); s++) {
        if (sets[s].size() <= 1) continue;
        uint64_t base = (sets[s][0].second >> align_bit) & mask_window;
        for (size_t j = 1; j < sets[s].size(); j++) {
            uint64_t v = ((sets[s][j].second >> align_bit) & mask_window) ^ base;
            if (v == 0) continue; // trivial equation
            rows.push_back(v);
        }
    }

    if (rows.empty()) return results;

    // Gaussian elimination (row-reduction) over GF(2)
    int m = rows.size();
    int row = 0;
    std::vector<int> pivot_col(B, -1);

    for (int col = 0; col < B && row < m; col++) {
        // find pivot row with bit 'col' set
        int sel = -1;
        for (int r = row; r < m; r++) {
            if ((rows[r] >> col) & 1ULL) { sel = r; break; }
        }
        if (sel == -1) continue;
        // swap
        std::swap(rows[row], rows[sel]);
        pivot_col[col] = row;
        // eliminate this bit from all other rows
        for (int r = 0; r < m; r++) {
            if (r != row && ((rows[r] >> col) & 1ULL)) {
                rows[r] ^= rows[row];
            }
        }
        row++;
    }

    int rank = row;
    int nullity = B - rank;
    logDebug("GF2 solve: B=%d, equations=%d, rank=%d, nullity=%d\n", B, m, rank, nullity);

    // Determine free columns (those not pivot)
    std::vector<int> free_cols;
    for (int c = 0; c < B; c++) {
        if (pivot_col[c] == -1) free_cols.push_back(c);
    }

    // If nullity == 0 then only zero solution -> no non-zero mask
    if (nullity == 0) return results;

    // Build nullspace basis vectors (t = nullity)
    // For each free column f, set x[f]=1 and compute pivot columns via rows
    std::vector<uint64_t> basis; basis.reserve(nullity);
    for (int fi = 0; fi < (int)free_cols.size(); fi++) {
        uint64_t vec = 0;
        int fcol = free_cols[fi];
        // set free bit
        vec |= (1ULL << fcol);
        // compute pivot bits: for each pivot column pcol, x[pcol] = rows[pivot_row][fcol]
        for (int pcol = 0; pcol < B; pcol++) {
            int prow = pivot_col[pcol];
            if (prow == -1) continue;
            // if entry at (prow, fcol) is 1 then pivot bit must be 1
            if ((rows[prow] >> fcol) & 1ULL) vec |= (1ULL << pcol);
        }
        basis.push_back(vec);
    }

    // Now enumerate combinations of basis vectors to find solutions with Hamming weight == bits
    int t = basis.size();
    const int ENUM_LIMIT = 22; // up to 2^22 ~ 4M combos; adjust as needed
    if (t <= ENUM_LIMIT) {
        uint64_t combos = (1ULL << t);
        for (uint64_t mask = 1; mask < combos; mask++) {
            uint64_t vec = 0;
            // combine basis
            uint64_t mm = mask;
            while (mm) {
                int b = __builtin_ctzll(mm);
                vec ^= basis[b];
                mm &= mm - 1;
            }
            if ((int)popcount64(vec) == bits) {
                // shift by g_start_bit to match existing code convention
                results.push_back((pointer) (vec << g_start_bit));
            }
        }
    } else {
        // Nullity too large to enumerate fully. Use heuristics:
        //  - include each single basis vector if weight matches
        //  - include some pairwise combinations (cap)
        const int PAIR_CAP = 20000;
        for (int i = 0; i < t; i++) {
            uint64_t v = basis[i];
            if ((int)popcount64(v) == bits) results.push_back((pointer) (v << g_start_bit));
        }
        int added = 0;
        for (int i = 0; i < t && added < PAIR_CAP; i++) {
            for (int j = i + 1; j < t && added < PAIR_CAP; j++) {
                uint64_t v = basis[i] ^ basis[j];
                if ((int)popcount64(v) == bits) {
                    results.push_back((pointer) (v << g_start_bit));
                    added++;
                }
            }
        }
        // If still empty, include low-weight basis combinations by greedy
        if (results.empty()) {
            // greedy: try combinations formed by XORing up to 4 smallest-weight basis vectors
            std::vector<std::pair<int,int>> weights; weights.reserve(t);
            for (int i = 0; i < t; i++) weights.emplace_back(popcount64(basis[i]), i);
            std::sort(weights.begin(), weights.end());
            int limit = std::min(t, 12);
            for (int i = 0; i < limit; i++) {
                for (int j = i; j < limit; j++) {
                    for (int k = j; k < limit; k++) {
                        uint64_t v = basis[weights[i].second] ^ basis[weights[j].second] ^ basis[weights[k].second];
                        if ((int)popcount64(v) == bits) results.push_back((pointer) (v << g_start_bit));
                    }
                }
            }
        }
    }

    // Deduplicate results
    std::sort(results.begin(), results.end());
    results.erase(std::unique(results.begin(), results.end()), results.end());
    return results;
}


// ----------------------------------------------
void save_bank_functions(const char* filename,
                        const std::map<int, std::vector<pointer>>& functions,
                        const std::vector<pointer>& false_positives,
                        const std::map<int, std::vector<pointer>>& duplicates,
                        const std::map<int, std::vector<double>>& prob) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file %s for writing\n", filename);
        return;
    }

    // Save found functions
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        for (int i = 0; i < functions.at(bits).size(); i++) {
            bool show = true;
    
            // Skip false positives
            for (int fp = 0; fp < false_positives.size(); fp++) {
                if ((functions.at(bits)[i] & false_positives[fp]) == false_positives[fp]) {
                    show = false;
                    break;
                }
            }
            if (!show) continue;

            // Skip duplicates
            for (int dup = 0; dup < duplicates.at(bits).size(); dup++) {
                if (functions.at(bits)[i] == duplicates.at(bits)[dup]) {
                    show = false;
                    break;
                }
            }
            if (!show) continue;

            // Write the function to file
            // Each line contains the physical address bits that are XORed
            char* bit_string = name_bits(functions.at(bits)[i]);
            // Remove trailing space
            if (strlen(bit_string) > 0 && bit_string[strlen(bit_string)-1] == ' ') {
                bit_string[strlen(bit_string)-1] = '\0';
            }
            fprintf(fp, "%s\n", bit_string);
        }
    }

    fclose(fp);
    printf("Bank mapping functions saved to %s\n", filename);
}

// ----------------------------------------------
int main(int argc, char *argv[]) {
    size_t tries, t;
    std::set <addrpair> addr_pool;
    std::map<int, std::list<addrpair> > timing;
    size_t hist[MAX_HIST_SIZE];
    int c;
    int samebank_threshold = -1;
    int cpu_affinity = -1;
    int quit_sets = -1; // number of sets to quit early

    while ((c = getopt(argc, argv, "b:c:e:r:g:m:i:j:s:q:t:v:f:B:K:")) != EOF) {
        switch (c) {
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
        case 's':
            expected_sets = atoi(optarg);
            break;
        case 'q':
            quit_sets = atoi(optarg);
            break;
        case 't':
            samebank_threshold = atoi(optarg);
            break;
        case 'v':
            verbosity = atoi(optarg);
            break;
        case 'f':
            g_output_file = optarg;
            break;
        case 'B':
            bg_samples_per_pair = atoi(optarg);
            if (bg_samples_per_pair <= 0) bg_samples_per_pair = 1;
            break;
        case 'K':
            enable_bg_detection = atoi(optarg);
            break;
        case ':':
            printf("Missing option.\n");
            exit(1);
            break;
        default:
            printf(
                "Usage %s [-m <memory size in MB> | -g <memory size in GB>] [-i <number of outer loops>] [-j <number of inner loops>] [-s <expected sets>] [-q <sets for early quit>] [-t <threshold cycles>] [-f <output file>] [-B <BG samples per pair>] [-K <enable BG detection 0|1>]\n",
                argv[0]);
            exit(0);
            break;
        }
    }

    // check mapping size
    // if (mapping_size > getPhysicalMemorySize() / 2) {
    //     logWarning("Mapping size is too large, reducing to %zu MB\n",
    //                getPhysicalMemorySize() / 2 / 1024 / 1024);
    //     mapping_size = getPhysicalMemorySize() / 2;
    // }

    srand(time(NULL));
    g_page_size = sysconf(_SC_PAGESIZE);
    initPagemap();
    setupMapping();

    // print the commandline used
    printf("Command:");
    for (int i = 0; i < argc; i++) {
        printf(" %s", argv[i]);
    }
    printf("\n");

    logInfo("Mapping has %zu MB\n", mapping_size / 1024 / 1024);

    logDebug("CPU: %s\n", getCPUModel());
    logDebug("Number of reads: %lu x %lu\n", num_reads_outer, num_reads_inner)
    logDebug("Expected sets: %lu\n", expected_sets);
    printf("Bank-group detection: %s (samples per pair=%d)\n",
        enable_bg_detection ? "enabled" : "disabled",
        bg_samples_per_pair);

    // affinity to core cpu_affinity
    if (cpu_affinity < 0) {
        cpu_affinity = 0;
    }
    logInfo("Setting CPU affinity to core %d\n", cpu_affinity);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_affinity, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) != 0) {
        perror("sched_setaffinity");
        exit(1);
    }
    // set high priority
    int prio = -20;
    logInfo("Setting priority to %d\n", prio);
    if (setpriority(PRIO_PROCESS, 0, prio) != 0) {
        perror("setpriority");
        exit(1);
    }

    pointer first, second;
    pointer first_phys, second_phys;
    pointer base, base_phys;

    int found_sets = 0;
    int found_siblings = 0;

    tries = expected_sets * 125; // DEBUG: original 125.

    // build address pool
    while (int cur_count = addr_pool.size() < tries) {
        getRandomAddress(&second, &second_phys);
        addr_pool.insert(std::make_pair(second, second_phys));
        // if (cur_count != addr_pool.size())
        //     logDebug("addr_pool[%ld]: 0x%0lx\n", addr_pool.size(), second_phys);
    }

    auto ait = addr_pool.begin();
    // std::advance(ait, rand() % addr_pool.size());
    base = ait->first;
    base_phys = ait->second;

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


    // row hit timing
    t = getTiming(base, base + 64);
    logInfo("Average ROW hit cycles: %ld \n", t);

    int failed;
    uint64_t measure_count = 0;

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

        // choose base address from remaining addresses
        auto ait = addr_pool.begin();
        std::advance(ait, rand() % addr_pool.size());
        base = ait->first;
        base_phys = ait->second;
        addr_pool.erase(ait);

        logInfo("Searching for set %d (try %d): base_phy=0x%lx\n",
                found_sets + 1, failed, base_phys);
        timing.clear();

        auto pool_it = addr_pool.begin();
        // iterate over the addr_pool and measure access times
        measure_count = 0;
        while (pool_it != addr_pool.end()) {

            first = pool_it->first;
            first_phys = pool_it->second;

            // measure timing
            t = getTiming(base, first);
            measure_count++;

            // sched_yield();
            // store virtual+physical pair: (virtual, physical)
            timing[t].push_back(std::make_pair(first, first_phys));

            // advance iterator
            pool_it++;
        }
        printf("(%lu)\n", measure_count);

        // identify sets -> must be on the right, separated in the histogram
        std::vector <addrpair> new_set; // virtual and physical address pairs of the identified set
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
                double distance1 = 0, distance2 = 0; // average distance to cluster center
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
                    int64_t dis1 = 0, dis2 = 0;

                    for (int b = min; b <= max; b++) {
                        size_t c = hist[b];
                        if (c == 0) continue;
                        double d1 = fabs((double)b - cluster1);
                        double d2 = fabs((double)b - cluster2);
                        if (d1 < d2) {
                            sum1 += c * b;
                            cnt1 += c;
                            dis1 += c * d1;
                        } else {
                            sum2 += c * b;
                            cnt2 += c;
                            dis2 += c * d2;
                        }
                    }

                    if (cnt1 > 0) {
                        cluster1 = (double)sum1 / cnt1;
                        distance1 = (double)dis1 / cnt1;
                    }
                    if (cnt2 > 0) {
                        cluster2 = (double)sum2 / cnt2;
                        distance2 = (double)dis2 / cnt2;
                    }

                    iterations++;

                    logDebug("K-means iteration %d: cluster1: %.2f (+- %.2f) cluster2: %.2f (+- %.2f)\n",
                             iterations, cluster1, distance1, cluster2, distance2);
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

        new_set.push_back(std::make_pair(base, base_phys)); // this is needed. another bug in the original code

        // remove found addresses from pool
        for (hit = timing.begin(); hit != timing.end(); hit++) {
            if (hit->first >= found && hit->first <= max) {
                for (std::list<addrpair>::iterator it = hit->second.begin();
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
            t = getTiming(base, new_set[i].first);
            if (t < found - 2) { // within 2 cycles of the threshold is okay
                logWarning("Validation failed: address 0x%lx has timing %lu < %d. removing it from the set\n",
                           new_set[i].first, t, found);
                // remove it from the new set
                new_set.erase(new_set.begin() + i);
                i--;
            }
        }
        logDebug("Validation done. New set size: %lu\n", new_set.size());
        // remove found addresses from pool
        logDebug("Removing found addresses from pool (%lu) -->", addr_pool.size());
        for (auto it = addr_pool.begin(); it != addr_pool.end();) {
            int erased = 0;
            for (auto nit = new_set.begin(); nit != new_set.end(); nit++) {
                if (*nit == *it) {
                    it = addr_pool.erase(it);
                    erased = 1;
                    break;
                }
            }
            if (!erased) {
                it++;
            }
        }
        logDebug(" %lu\n", addr_pool.size());

        // save identified set if one was found
        sets.push_back(new_set);
        found_siblings += new_set.size();

        found_sets++;
        if (quit_sets > 0 && found_sets >= quit_sets) {
            logInfo("Quitting early after finding %d sets as requested.\n", quit_sets);
            break;
        }
    }
    logDebug("Done measuring. found_sets: %d found_siblings: %d\n",
             found_sets, found_siblings);

    for (int set = 0; set < sets.size(); set++) {
        logInfo("Set %d: count: %ld\n",
                set + 1, sets[set].size());
        char filename[100];
        sprintf(filename, "set%d.txt", set + 1);
        FILE *f = fopen(filename, "w"); // overwrite if file exists
        if (!f) {
            logWarning("Cannot open file %s for writing\n", filename);
            continue;
        }
        for (int j = 0; j < sets[set].size(); j++) {
            fprintf(f, "  0x%lx\n", sets[set][j].second);
        }
        fclose(f);
    }
    
    // try to find a xor function
    std::map<int, std::vector<double> > prob;
    for (int bits = 1; bits <= MAX_XOR_BITS; bits++) {
        // Use linear GF(2) solver instead of expensive combinatorial enumeration
        functions[bits] = find_function_linear(bits, POINTER_SIZE, g_start_bit);
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
                matrix[r][cols - b - 1] = (f & (1ull << b)) ? 1 : 0;
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

    std::vector<BGFunctionCandidate> bg_candidates;
    if (enable_bg_detection) {
        auto pair_timings = collect_pair_timings(sets, bg_samples_per_pair);
        if (!pair_timings.empty()) {
            BGThresholdInfo bg_info = compute_bg_threshold(pair_timings);
            if (bg_info.valid) {
                printf("BG timing clusters: low=%.2f high=%.2f threshold=%.2f\n",
                       bg_info.low_center, bg_info.high_center, bg_info.threshold);
                auto bg_labels = derive_bg_labels(pair_timings, bg_info, sets.size());
                printf("BG labels per set: ");
                std::map<int, int> label_hist;
                for (size_t idx = 0; idx < bg_labels.size(); idx++) {
                    printf("%zu:%d ", idx + 1, bg_labels[idx]);
                    label_hist[bg_labels[idx]]++;
                }
                printf("\n");
                printf("BG label histogram: ");
                for (const auto& entry : label_hist) {
                    printf("%d=>%d ", entry.first, entry.second);
                }
                printf("\n");
                bg_candidates = rank_bg_functions(bg_labels, sets, functions,
                                                  false_positives, duplicates);
                print_bg_candidates(bg_candidates);
            } else {
                logWarning("%s", "Bank-group detection skipped: could not find distinct timing clusters.\n");
            }
        } else {
            logWarning("%s", "Bank-group detection skipped: insufficient inter-set timing samples.\n");
        }
    } else {
        printf("Bank-group detection disabled (use -K 1 to enable).\n");
    }

    const char* output_filename = g_output_file ? g_output_file : "map.txt";


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

    // Save bank mapping functions to file
    save_bank_functions(output_filename, functions, false_positives, duplicates, prob);
    save_bank_group_functions(output_filename, bg_candidates);
    
    fprintf(stderr, "Finishing\n");
    exit(1);
    return 0;

}
