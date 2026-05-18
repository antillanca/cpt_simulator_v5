import sys
import torch

def compare_dicts(d1, d2, path=""):
    diffs = []
    all_keys = set(d1.keys()) | set(d2.keys())
    for k in sorted(all_keys):
        p = f"{path}.{k}" if path else k
        if k not in d1:
            diffs.append((p, "Missing in a", d2[k]))
        elif k not in d2:
            diffs.append((p, d1[k], "Missing in b"))
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            diffs.extend(compare_dicts(d1[k], d2[k], p))
        elif d1[k] != d2[k]:
            diffs.append((p, d1[k], d2[k]))
    return diffs

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_checkpoints.py <ckpt_a> <ckpt_b>")
        return 1

    a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)

    print("=== Non-nested Top-level Fields ===")
    for k in sorted(a.keys()):
        if k in ["state_dict", "optimizer_state", "extra"]:
            continue
        if a[k] != b.get(k):
            print(f"  {k}:")
            print(f"    a: {a[k]}")
            print(f"    b: {b.get(k)}")

    print("\n=== Nested Extra Fields ===")
    extra_diffs = compare_dicts(a.get("extra", {}), b.get("extra", {}))
    for p, val_a, val_b in extra_diffs:
        print(f"  {p}:")
        print(f"    a: {val_a}")
        print(f"    b: {val_b}")

if __name__ == "__main__":
    main()
