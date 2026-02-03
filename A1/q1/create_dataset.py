import sys
import random


def parse_item_rng(s):
    a, b = map(int, s.split('-'))
    return list(range(a, b + 1))


def create_syn_dataset(univ, n, out):
    import random

    CORE_SIZE = 30      
    CORE_SUPPORT = 0.60  

    MIN_CORE_PICK = 20    
    MAX_CORE_PICK = 28   

    MIN_BG = 5
    MAX_BG = 12

    assert MAX_CORE_PICK <= CORE_SIZE

    core = random.sample(univ, CORE_SIZE)
    rest = [x for x in univ if x not in core]

    num_core = int(n * CORE_SUPPORT)

    with open(out, "w") as f:

        # balanced dense-ish core 
        for _ in range(num_core):
            k = random.randint(MIN_CORE_PICK, MAX_CORE_PICK)
            txn = random.sample(core, k)
            txn += random.sample(rest, random.randint(2, 5))
            f.write(" ".join(map(str, sorted(txn))) + "\n")

        # background 
        for _ in range(n - num_core):
            size = random.randint(MIN_BG, MAX_BG)
            txn = random.sample(univ, size)
            f.write(" ".join(map(str, sorted(txn))) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_dataset.py <item_range> <num_txns> <output_file>")
        sys.exit(1)

    univ = parse_item_rng(sys.argv[1])
    create_syn_dataset(univ, int(sys.argv[2]), sys.argv[3])