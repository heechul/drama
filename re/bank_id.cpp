#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstring>

int g_debug = 0;
static std::vector<std::vector<int>> g_bank_functions;

// Read bank bit mapping functions from file
void read_bank_map_file(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open map file %s\n", filename);
        exit(1);
    }

    char line[256];
    g_bank_functions.clear();

    while (fgets(line, sizeof(line), fp)) {
        // Skip empty lines and comments
        if (line[0] == '\n' || line[0] == '#') continue;
        std::vector<int> function_bits;
        char* token = strtok(line, " \t\n");
        while (token != nullptr) {
            int bit = atoi(token);
            function_bits.push_back(bit);
            token = strtok(nullptr, " \t\n");
        }
        if (!function_bits.empty()) {
            g_bank_functions.push_back(function_bits);
        }
    }
    fclose(fp);

    if (g_debug) {
        printf("Loaded %zu bank mapping functions:\n", g_bank_functions.size());
        for (size_t i = 0; i < g_bank_functions.size(); i++) {
            printf("Function %zu: XOR bits ", i);
            for (int bit : g_bank_functions[i]) {
                printf("%d ", bit);
            }
            printf("\n");
        }
    }
}

int paddr_to_color(unsigned long mask, unsigned long paddr)
{
    int color = 0;
    for (size_t func_idx = 0; func_idx < g_bank_functions.size(); func_idx++) {
        int bit_result = 0;
        
        // XOR all the specified bits for this function
        for (int bit_pos : g_bank_functions[func_idx]) {
            bit_result ^= ((paddr >> bit_pos) & 0x1);
        }
        
        // Set the corresponding bit in the color
        if (bit_result) {
            color |= (1 << func_idx);
        }
    }
    return color;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <map file> <physical_address>" << std::endl;
        return 1;
    }
    uint64_t address = std::stoull(argv[2], nullptr, 0);

    read_bank_map_file(argv[1]);

    int bank_num = 0;

    bank_num = paddr_to_color(0, address);
    std::cout << "Physical address 0x" << std::hex << address << " maps to bank " << std::dec << bank_num << std::endl;
    
    return 0;
}