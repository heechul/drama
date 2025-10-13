#!/usr/bin/env python3
import glob
import os
import re
from typing import List, Tuple


def parse_map_file(path: str) -> List[List[int]]:
    lines = []
    with open(path, 'r') as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            if s.startswith('bank_bit'):
                # skip constant lines
                continue
            parts = s.split()
            bits = []
            for p in parts:
                try:
                    bits.append(int(p))
                except ValueError:
                    pass
            if bits:
                lines.append(bits)
    return lines


def read_set_files(dirpath: str) -> List[List[int]]:
    # find set*.txt and sort by the numeric suffix
    files = glob.glob(os.path.join(dirpath, 'set*.txt'))
    def keyfn(p):
        m = re.search(r'set(\d+)\.txt$', p)
        return int(m.group(1)) if m else p
    files.sort(key=keyfn)
    banks = []
    for p in files:
        addrs = []
        with open(p, 'r') as f:
            for ln in f:
                s = ln.strip().split()[0] if ln.strip() else ''
                if not s: continue
                try:
                    if s.startswith('0x') or any(c in s.lower() for c in 'abcdef'):
                        addrs.append(int(s, 16))
                    else:
                        addrs.append(int(s, 10))
                except Exception:
                    continue
        if addrs:
            banks.append(addrs)
    return banks


def compute_code(addr: int, mapping: List[List[int]]) -> Tuple[int]:
    # mapping is list of bit-index lists; compute tuple of bits (MSB to LSB as order)
    code = []
    for bits in mapping:
        s = 0
        for b in bits:
            s ^= ((addr >> b) & 1)
        code.append(s)
    return tuple(code)


def mat_solve_gf2(A: List[List[int]], b: List[int]) -> Tuple[bool, List[int]]:
    # Solve A x = b over GF(2). A: m x n. Return (solvable, one_solution_of_length_n)
    if not A:
        # 0 equations: any x works; return zero vector
        return True, [0]* (0)
    m = len(A); n = len(A[0])
    # build augmented matrix
    aug = [row[:] + [bv] for row,bv in zip(A, b)]
    row = 0
    pivcols = [-1]*m
    for col in range(n):
        sel = None
        for r in range(row, m):
            if aug[r][col]:
                sel = r; break
        if sel is None: continue
        aug[row], aug[sel] = aug[sel], aug[row]
        pivcols[row] = col
        # eliminate
        for r in range(m):
            if r != row and aug[r][col]:
                for c in range(col, n+1):
                    aug[r][c] ^= aug[row][c]
        row += 1
        if row == m: break
    # check consistency: any all-zero row in A but aug rhs 1 => no solution
    for r in range(row, m):
        if all(c==0 for c in aug[r][:-1]) and aug[r][-1]:
            return False, []
    # construct a solution: set free vars to 0, compute pivots
    x = [0]*n
    for r in range(row-1, -1, -1):
        pc = pivcols[r]
        if pc == -1: continue
        s = aug[r][-1]
        for c in range(pc+1, n):
            if aug[r][c] and x[c]:
                s ^= 1
        x[pc] = s
    return True, x


def try_find_transform(RC: List[Tuple[int]], RP: List[Tuple[int]]) -> Tuple[bool, List[List[int]]]:
    # RC and RP are lists of length B of k-bit tuples. Find kxk matrix M such that for all rows: RC_row * M = RP_row
    B = len(RC)
    if B == 0: return False, []
    k = len(RC[0])
    for row in RC:
        if len(row) != k: return False, []
    for row in RP:
        if len(row) != k: return False, []
    # Build A as B x k matrix RC, but we need to solve for each column of M
    # For each j in 0..k-1, solve RC * m_j = RP_col_j
    A = [list(r) for r in RC]
    M = [[0]*k for _ in range(k)]
    for j in range(k):
        b = [rp[j] for rp in RP]
        solvable, x = mat_solve_gf2(A, b)
        if not solvable:
            return False, []
        # x length is k
        for i in range(k):
            M[i][j] = x[i]
    # Verify invertibility? We can check determinant (rank) of M
    # compute rank of M
    # convert M to k x k; compute rank
    # We'll just check M is invertible by computing rank
    # Build Mcols? For now return M
    return True, M


def apply_transform(code: Tuple[int], M: List[List[int]]) -> Tuple[int]:
    k = len(code)
    out = [0]*k
    for j in range(k):
        s = 0
        for i in range(k):
            if code[i] and M[i][j]: s ^= 1
        out[j] = s
    return tuple(out)


def main():
    dirpath = os.path.dirname(__file__)
    map1 = parse_map_file(os.path.join(dirpath, 'map.txt'))
    map2 = parse_map_file(os.path.join(dirpath, 'recovered_bank_mapping.txt'))
    print('map1 k=', len(map1), 'map2 k=', len(map2))
    banks = read_set_files(dirpath)
    print('Found', len(banks), 'bank files')
    # Build list of addresses with bank ids
    addrs = []
    for bi, rows in enumerate(banks):
        for a in rows:
            addrs.append((bi, a))
    mismatches = 0
    diff_examples = []
    for bi, a in addrs:
        c1 = compute_code(a, map1)
        c2 = compute_code(a, map2)
        if c1 != c2:
            mismatches += 1
            if len(diff_examples) < 10:
                diff_examples.append((bi, hex(a), c1, c2))
    print('Total addrs:', len(addrs))
    print('Mismatches:', mismatches)
    if mismatches > 0:
        print('Examples:')
        for ex in diff_examples:
            print(ex)
    # Now try transform between bank codes using representatives (first address per bank)
    reps = [rows[0] for rows in banks]
    RC = [compute_code(a, map1) for a in reps]
    RP = [compute_code(a, map2) for a in reps]
    ok, M = try_find_transform(RC, RP)
    print('Found linear transform mapping map1->map2:', ok)
    if ok:
        print('Matrix M (k x k):')
        for row in M:
            print(''.join(str(x) for x in row))
        # verify on all addresses
        bad = 0
        for bi, a in addrs:
            c1 = compute_code(a, map1)
            c1t = apply_transform(c1, M)
            c2 = compute_code(a, map2)
            if c1t != c2:
                bad += 1
        print('After transform, mismatches:', bad)

if __name__ == '__main__':
    main()
