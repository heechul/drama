#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GF(2) DRAM bank-mapping solver from address sets.

Inputs: one text file per bank (distinct bank), each line = physical address
(hex '0x..' or hex '...' or decimal). Blank lines & lines starting with '#'
are ignored.

This script:
  1) Builds a GF(2) linear system so that all addresses in the same file map
     to the SAME k-bit code. It does not rely on your file numbering scheme.
  2) Solves constraints D * X = 0 to find X in the nullspace that keeps
     same-bank addresses collapsed to one code.
  3) Projects representatives through X and selects a minimal set of output
     bits that uniquely identify each bank.
  4) Prints bank-bit = XOR of PA bits, and verifies on all input addresses.

Assumptions:
  - The true mapping from PA bits to bank bits is linear over GF(2)
    (common on many platforms).
  - The chosen bit window [lowbit, highbit] covers all relevant PA bits that
    feed the bank function.

Heechul: this is the label-free variant, robust to arbitrary file ordering.
"""

import argparse
import math
import sys
from typing import List, Tuple

# -------------------------- GF(2) helpers --------------------------

def parse_addr(s: str) -> int:
    s = s.strip()
    if not s or s.startswith('#'):
        return None  # sentinel
    try:
        if s.startswith('0x') or s.startswith('0X'):
            return int(s, 16)
        # hex without 0x?
        if any(c in s.lower() for c in 'abcdef'):
            return int(s, 16)
        # decimal
        return int(s, 10)
    except ValueError:
        return None

def read_addr_file(path: str) -> List[int]:
    out = []
    with open(path, 'r') as f:
        for line in f:
            v = parse_addr(line)
            if v is None:
                continue
            out.append(v)
    if not out:
        raise ValueError(f"No valid addresses in {path}")
    return out

def build_bitvec(addr: int, lowbit: int, highbit: int) -> List[int]:
    return [ (addr >> b) & 1 for b in range(lowbit, highbit+1) ]  # length n

def mat_rank_gf2(M: List[List[int]]) -> int:
    """Return rank over GF(2), mutating a local copy."""
    if not M: return 0
    A = [row[:] for row in M]
    n_rows, n_cols = len(A), len(A[0])
    r = 0
    for c in range(n_cols):
        pivot = None
        for i in range(r, n_rows):
            if A[i][c]:
                pivot = i; break
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        for i in range(n_rows):
            if i != r and A[i][c]:
                # row_i ^= row_r
                rowi, rowr = A[i], A[r]
                for j in range(c, n_cols):
                    rowi[j] ^= rowr[j]
        r += 1
        if r == n_rows: break
    return r

def nullspace_basis_gf2(D: List[List[int]]) -> List[List[int]]:
    """
    Compute a basis N for the right nullspace of D (D * x = 0).
    Returns N as a list of 's' basis column vectors of length n (so N is n x s).
    """
    if not D:
        # Nullspace is entire space; return identity basis (we'll decide n later)
        return None

    m = len(D)
    n = len(D[0])
    A = [row[:] for row in D]  # m x n
    pivcol = [-1] * m
    row = 0
    for col in range(n):
        sel = -1
        for r in range(row, m):
            if A[r][col]:
                sel = r; break
        if sel == -1:
            continue
        A[row], A[sel] = A[sel], A[row]
        pivcol[row] = col
        # eliminate
        for r in range(m):
            if r != row and A[r][col]:
                # row_r ^= row_row
                for c in range(col, n):
                    A[r][c] ^= A[row][c]
        row += 1
        if row == m:
            break

    pivot_cols = set(c for c in pivcol if c != -1)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    # For each free var f, set x_f = 1, solve pivot vars
    for f in free_cols:
        x = [0]*n
        x[f] = 1
        # Back-substitute pivot vars (from last pivot row to first)
        for r in reversed(range(m)):
            pc = pivcol[r]
            if pc == -1:
                continue
            # sum over columns > pc (where A[r][c] could be 1)
            s = 0
            # Note: we only need to consider free columns and pivot columns > pc
            for c in range(pc+1, n):
                if A[r][c] and x[c]:
                    s ^= 1
            x[pc] = s  # because A[r][pc] is 1 in row-echelon form
        basis.append(x)
    # Edge case: full column rank -> nullspace only {0}; return empty basis
    return basis  # list of n-vectors (columns)

def matmul_gf2(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """A (m x n) times B (n x p) -> (m x p) over GF(2)."""
    if not A or not B:
        return []
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2
    C = [[0]*p for _ in range(m)]
    # Dense but fine for moderate sizes
    for i in range(m):
        Ai = A[i]
        one_cols = [c for c,v in enumerate(Ai) if v]
        for j in range(p):
            s = 0
            for c in one_cols:
                s ^= B[c][j]
            C[i][j] = s
    return C

def select_unique_columns(Z: List[List[int]], need: int) -> Tuple[List[int], int]:
    """
    Given Z (B x s), greedily pick up to 'need' columns that (a) increase rank and
    (b) try to make all B rows unique. Returns (picked_cols, rank_after).
    """
    if not Z: return ([], 0)
    B = len(Z); s = len(Z[0])
    picked = []
    current = []
    best_uniques = len(set())  # 0
    best_cols = []

    def rows_to_tuples(M):
        return tuple(tuple(row) for row in M)

    rank = 0
    for c in range(s):
        trial_cols = picked + [c]
        M = [[row[j] for j in trial_cols] for row in Z]  # B x len(trial_cols)
        r = mat_rank_gf2([row[:] for row in M])
        if r > rank:
            picked = trial_cols
            rank = r
            # track uniqueness
            uniques = len(set(rows_to_tuples(M)))
            if uniques > best_uniques:
                best_uniques = uniques
                best_cols = picked[:]
            if len(picked) == need:
                break

    if best_uniques < B and rank >= need:
        # Try to improve uniqueness by adding more columns if available
        for c in range(s):
            if c in picked: continue
            trial_cols = picked + [c]
            M = [[row[j] for j in trial_cols] for row in Z]
            uniques = len(set(rows_to_tuples(M)))
            if uniques > best_uniques:
                best_uniques = uniques
                best_cols = trial_cols[:]
            if best_uniques == B:
                picked = best_cols
                break

    if best_uniques == B:
        return (picked if picked else best_cols, rank)
    # Fallback to whatever increased rank the most
    return (picked if picked else best_cols, rank)

# -------------------------- Basis utilities --------------------------

ENUM_LIMIT = 16

def popcount(x: int) -> int:
    return x.bit_count()

def bits_from_mask(mask: int, lowbit: int) -> List[int]:
    bits = []
    bit = 0
    while mask:
        if mask & 1:
            bits.append(lowbit + bit)
        mask >>= 1
        bit += 1
    return bits

def row_reduce_masks(rows: List[int], nbits: int) -> List[int]:
    rows = [r for r in rows if r]
    rows = rows[:]
    r = 0
    for col in range(nbits - 1, -1, -1):
        piv = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        r += 1
    return [x for x in rows if x]

def matrix_to_masks(X: List[List[int]]) -> List[int]:
    if not X:
        return []
    n = len(X)
    k = len(X[0]) if X[0] else 0
    masks = []
    for j in range(k):
        mask = 0
        for i in range(n):
            if X[i][j]:
                mask |= (1 << i)
        masks.append(mask)
    return masks

def masks_to_matrix(masks: List[int], n: int) -> List[List[int]]:
    k = len(masks)
    X = [[0] * k for _ in range(n)]
    for j, mask in enumerate(masks):
        for i in range(n):
            if (mask >> i) & 1:
                X[i][j] = 1
    return X

def select_compact_masks(masks: List[int], nbits: int, lowbit: int) -> List[int]:
    if not masks:
        return []
    basis = row_reduce_masks(masks, nbits)
    if not basis:
        return []
    chosen = basis[:]
    if len(basis) <= ENUM_LIMIT:
        combos = []
        n = len(basis)
        for combo in range(1, 1 << n):
            vec = 0
            for idx in range(n):
                if (combo >> idx) & 1:
                    vec ^= basis[idx]
            if vec:
                combos.append((popcount(vec), bits_from_mask(vec, lowbit), vec))
        combos.sort()

        selected = []
        selected_rr = []
        for _, _, vec in combos:
            new_rr = row_reduce_masks(selected_rr + [vec], nbits)
            if len(new_rr) > len(selected_rr):
                selected.append(vec)
                selected_rr = new_rr
                if len(selected) == len(basis):
                    break
        if len(selected) == len(basis):
            chosen = selected
    chosen.sort(key=lambda m: (popcount(m), bits_from_mask(m, lowbit)))
    return chosen

def apply_matrix_row(row_bits: List[int], X: List[List[int]]) -> Tuple[int, ...]:
    if not X:
        return tuple()
    n = len(X)
    k = len(X[0]) if X[0] else 0
    if k == 0:
        return tuple()
    one_cols = [i for i, v in enumerate(row_bits) if v]
    code = [0] * k
    for j in range(k):
        sbit = 0
        for c in one_cols:
            sbit ^= X[c][j]
        code[j] = sbit
    return tuple(code)

def compute_bank_codes(reps: List[List[int]], X: List[List[int]]) -> List[Tuple[int, ...]]:
    return [apply_matrix_row(rep, X) for rep in reps]

def verify_addresses(banks: List[List[List[int]]], X: List[List[int]], bank_codes: List[Tuple[int, ...]], verbose: bool = False) -> Tuple[int, int]:
    mismatches = 0
    total = 0
    for b, rows in enumerate(banks):
        code_b = bank_codes[b]
        for r in rows:
            total += 1
            code = apply_matrix_row(r, X)
            if code != code_b:
                mismatches += 1
                if verbose:
                    print(f"Mismatch in bank {b}: got {code}, expected {code_b}")
    return mismatches, total

def remove_constant_bank_bits(X: List[List[int]], banks: List[List[List[int]]]) -> Tuple[List[List[int]], int]:
    if not X:
        return X, 0
    n = len(X)
    k = len(X[0]) if X[0] else 0
    if k == 0:
        return X, 0
    total = sum(len(rows) for rows in banks)
    col_sums = [0] * k
    for rows in banks:
        for r in rows:
            one_cols = [i for i, v in enumerate(r) if v]
            for j in range(k):
                sbit = 0
                for c in one_cols:
                    sbit ^= X[c][j]
                col_sums[j] += sbit
    nonconst_cols = [j for j in range(k) if 0 < col_sums[j] < total]
    if len(nonconst_cols) == k:
        return X, 0
    new_X = [[X[i][j] for j in nonconst_cols] for i in range(n)]
    removed = k - len(nonconst_cols)
    return new_X, removed

def compact_matrix_columns(X: List[List[int]], nbits: int, lowbit: int) -> List[List[int]]:
    if not X:
        return X
    k = len(X[0]) if X[0] else 0
    if k == 0:
        return X
    masks = matrix_to_masks(X)
    compact_masks = select_compact_masks(masks, nbits, lowbit)
    if not compact_masks:
        return X
    return masks_to_matrix(compact_masks, nbits)

# -------------------------- Main solver --------------------------

def main():
    ap = argparse.ArgumentParser(description="GF(2) bank mapping solver from per-bank address files.")
    ap.add_argument('--files', nargs='+', required=True,
                    help='One file per bank (each contains PAs for that bank).')
    ap.add_argument('--lowbit', type=int, default=5,
                    help='Lowest PA bit to include (default: 5 to skip 32B line).')
    ap.add_argument('--highbit', type=int, default=40,
                    help='Highest PA bit to include (inclusive).')
    ap.add_argument('--verbose', action='store_true', help='Extra prints.')
    args = ap.parse_args()

    bank_files = args.files
    lowbit, highbit = args.lowbit, args.highbit
    if highbit < lowbit:
        print("highbit must be >= lowbit", file=sys.stderr)
        sys.exit(1)

    # Load addresses per bank, build rows of PA bits
    banks = []
    for path in bank_files:
        addrs = read_addr_file(path)
        rows = [build_bitvec(a, lowbit, highbit) for a in addrs]
        banks.append(rows)

    total_addresses = sum(len(rows) for rows in banks)

    B = len(banks)
    n = highbit - lowbit + 1
    if args.verbose:
        print(f"Loaded {B} banks; using PA bits a{lowbit}..a{highbit} (n={n})")

    # Build constraints: for each bank b, all rows must map to SAME code.
    # Let r_b be the first row (representative). For any row x in bank b,
    # (x XOR r_b) * X = 0  ->  (x - r_b) over GF(2)
    D = []             # constraint matrix (t x n)
    reps = []          # representatives (B x n)
    total_rows = 0

    for b, rows in enumerate(banks):
        reps.append(rows[0][:])
        for i in range(1, len(rows)):
            diff = [rows[i][j] ^ rows[0][j] for j in range(n)]
            if any(diff):
                D.append(diff)
        total_rows += len(rows)

    # Find nullspace basis N of D (n x s)
    N = nullspace_basis_gf2(D)
    if N is None:
        # D was empty -> no constraints -> N = I (full space)
        N = [[1 if i == j else 0] for i in range(n) for j in range(n)]  # wrong shape
        # Fix shape: N should be n x n (identity)
        N = []
        for j in range(n):
            col = [0]*n
            col[j] = 1
            N.append(col)
        # N is list of n columns; we want N as n x s -> transpose our column list later as needed
    # Ensure N is n x s (columns)
    # Our nullspace_basis_gf2 returned a list of n-vectors (columns). Good.

    s = len(N)  # number of basis columns
    if s == 0:
        print("No nullspace: constraints force a unique (trivial) code; "
              "try widening bit range (--highbit).", file=sys.stderr)
        sys.exit(2)

    # Convert N (list of columns) to n x s matrix
    Nmat = [[N[col][row] for col in range(s)] for row in range(n)]  # n x s

    # Build R (B x n) from representatives, then Z = R * N (B x s)
    R = reps  # B x n
    Z = matmul_gf2(R, Nmat)  # B x s

    # Choose minimal k columns from Z that give unique bank codes
    need_k = max(math.ceil(math.log2(B)), 1) if B > 1 else 1
    picked_cols, rankZ = select_unique_columns(Z, need_k)
    if args.verbose:
        print(f"Rank(Z)={rankZ}, picked columns={picked_cols}")

    # If uniqueness not achieved, try more columns (if available)
    if len(picked_cols) < need_k or len({tuple(row[c] for c in picked_cols) for row in Z}) < B:
        # try to include more columns until rows are unique or we run out
        avail = [c for c in range(len(Z[0])) if c not in picked_cols]
        for c in avail:
            picked_cols.append(c)
            if len({tuple(row[c2] for c2 in picked_cols) for row in Z}) == B:
                break

    codes_unique = len({tuple(row[c] for c in picked_cols) for row in Z}) == B
    if not codes_unique:
        print("Warning: could not find a set of output bits that uniquely label all banks.\n"
              "Try widening [--lowbit, --highbit] or ensure your input sets are clean.",
              file=sys.stderr)

    k = len(picked_cols)
    # Compute X = N * W, where W selects the picked columns (i.e., W = column selector)
    # Simple: take W as s x k with columns equal to e_{picked_col}
    W = [[0]*k for _ in range(s)]
    for j, c in enumerate(picked_cols):
        if c < 0 or c >= s: continue
        W[c][j] = 1

    # X = Nmat (n x s) * W (s x k)  -> n x k
    X = matmul_gf2(Nmat, W)  # n x k

    X, removed_consts = remove_constant_bank_bits(X, banks)
    if removed_consts:
        print(f"\nNote: removed {removed_consts} constant bank bits (always 0 or 1).")

    X = compact_matrix_columns(X, n, lowbit)
    k = len(X[0]) if X and X[0] else 0
    if not X:
        print("\nNo bank-bit functions remain after compaction.")

    bank_codes = compute_bank_codes(reps, X)

    print("\n=== Verification on all input addresses ===")
    mismatches, total = verify_addresses(banks, X, bank_codes, args.verbose)
    if mismatches == 0:
        print(f"OK: all {total} addresses map consistently to their bank codes.")
    else:
        print(f"WARNING: {mismatches}/{total} addresses did not map to their bank code.")
        print("This often means the chosen bit window missed some true mapping bits,")
        print("or some addresses are mislabeled/noisy. Try increasing --highbit.")

    # Final report
    print("\n=== Final recovered bank-bit functions ===")
    for j in range(k):
        bits = [f"a{lowbit+i}" for i in range(n) if X[i][j] == 1]
        if bits:
            print(f"bank_bit[{j}] = {' ⊕ '.join(bits)}")
        else:
            print(f"bank_bit[{j}] = 0  (constant)")
    print("\n(You can copy-paste these functions for your use.)")
    print("\n=== End of report ===")

    # Optionally,
    # store recovered mapping into a file
    print("\nStoring recovered mapping to 'recovered_bank_mapping.txt'")
    with open("recovered_bank_mapping.txt", "w") as f:
        for j in range(k):
            bits = [f"{lowbit+i}" for i in range(n) if X[i][j] == 1]
            if bits:
                f.write(f"{' '.join(bits)}\n")
            else:
                f.write(f"bank_bit[{j}] = 0  (constant)\n")
if __name__ == '__main__':
    main()
