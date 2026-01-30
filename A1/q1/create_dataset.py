import sys
import random


def parse_item_rng(item_rng_in_str):
    """Parses a str "1-100" into a list of integers [1, 2, ..., 100]."""
    try:
        start, end = map(int, item_rng_in_str.split('-'))
        return list(range(start, end + 1))
    except ValueError:
        raise ValueError("Invalid item range format. Expected format: 'start-end'.")
    

def create_syn_dataset(item_univ, num_txns):
    """Creates a synthetic dataset of transactions"""
    op_fname = "dataset.dat"

    # Define 'poison' itemset - a frequent long itemset 
    if len(item_univ) < 20:
        raise ValueError("Item universe must be larger than 20 to create a distinct itemset.")
    
    poison_itemset = item_univ[:15]
    noise_itemset = item_univ[15:] # Random noise

    # Poison frequency. Cliff is between 50-90%
    poison_support = 0.60
    num_poison_txns = int(num_txns * poison_support)

    print(f"Generating {num_txns} transactions...")

    with open(op_fname, "w") as f:
        # Generate poison itemset transactions
        for _ in range(num_poison_txns):
            # Add noise to prevent identical transactions
            num_noise = random.randint(2, 5)
            txn = poison_itemset + random.sample(noise_itemset, num_noise)
            # Sort and save items
            f.write(" ".join(map(str, sorted(txn))) + "\n")

        # Generate normal transactions w/o poison itemset
        num_normal_txns = num_txns - num_poison_txns
        for _ in range(num_normal_txns):
            # Shorter & randomly sampled transactions
            txn_size = random.randint(5, 12)
            txn = random.sample(item_univ, txn_size)
            f.write(" ".join(map(str, sorted(txn))) + "\n")

    print(f"Synthetic dataset created and saved to {op_fname}")


if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_dataset.py <universal_itemset> <num_transactions>")
        sys.exit(1)

    item_rng_in_str = sys.argv[1]
    num_txns = int(sys.argv[2])

    try:
        univ = parse_item_rng(item_rng_in_str)
        create_syn_dataset(univ, num_txns)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)