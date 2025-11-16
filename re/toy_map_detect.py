# same_bank_nullspace.py
# Usage:
#   python3 same_bank_nullspace.py <<EOF
#   0x12345000
#   0x12347000
#   0x1ab45040
#   0x1ab47040
#   EOF

import sys

PA_BIT_LO = 6   # ignore 64B line bits
PA_BIT_HI = 33  # adjust if needed

def to_vec(pa):
    w = 0
    for b in range(PA_BIT_LO, PA_BIT_HI+1):
        if (pa >> b) & 1:
            w |= 1 << (b - PA_BIT_LO)
    return w

def row_reduce(rows):
    rows = [r for r in rows if r]
    r = 0
    nbits = PA_BIT_HI - PA_BIT_LO + 1
    rows = rows[:]
    for col in range(nbits-1, -1, -1):
        piv = None
        for i in range(r, len(rows)):
            if (rows[i] >> col) & 1:
                piv = i; break
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        r += 1
    return [x for x in rows if x]

def orthogonal_complement(basis):
    R = row_reduce(basis)
    nbits = PA_BIT_HI - PA_BIT_LO + 1
    rows = R[:]
    pivcol = []
    for r in rows:
        if r == 0: continue
        msb = r.bit_length()-1
        pivcol.append(msb)
    is_pivot = [0]*nbits
    for c in pivcol:
        if 0 <= c < nbits: is_pivot[c]=1
    free = [c for c in range(nbits) if not is_pivot[c]]

    null_basis = []
    for fc in free:
        vec = 1 << fc
        for r in rows:
            if r == 0: continue
            pc = r.bit_length()-1
            if (r >> fc) & 1:
                vec ^= (1 << pc)
        null_basis.append(vec)
    return row_reduce(null_basis)

def bits_of(mask):
    return [b for b in range(PA_BIT_LO, PA_BIT_HI+1) if (mask >> (b-PA_BIT_LO)) & 1]

def main():
    addrs = []
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        addrs.append(int(line, 0))
    if len(addrs) < 2:
        print("Need >=2 addresses from the SAME bank.")
        return
    ref = addrs[0]
    diffs = []
    for pa in addrs[1:]:
        d = to_vec(pa ^ ref)
        if d: diffs.append(d)
    if not diffs:
        print("No differences within bit window; widen PA_BIT_[LO,HI] or add more addresses.")
        return
    const_masks = orthogonal_complement(diffs)

    # remove false positive bank bit that is always 0 or 1 across all addresses
    filtered_masks = []
    for m in const_masks:
        is_const = True
        for b in bits_of(m):
            bit_val = (ref >> b) & 1
            for pa in addrs[1:]:
                if ((pa >> b) & 1) != bit_val:
                    is_const = False
                    break
            if not is_const:
                break
        if not is_const:
            filtered_masks.append(m)

    if not filtered_masks:
        print("No nontrivial constant masks found.")
        return
    for m in filtered_masks:
        print(" ".join(f"{b}" for b in bits_of(m)))

if __name__ == "__main__":
    main()
