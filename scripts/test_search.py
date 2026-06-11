# test_search.py — Search Module test suite
# Usage: cd <project_root> && python scripts/test_search.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES
from src.search_module import find_reachable_categories, find_popular_categories

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

def check(label, condition, detail=""):
    print(f"  [{PASS if condition else FAIL}] {label}" + (f" -- {detail}" if detail else ""))
    return condition

def test_reachable():
    print("\n=== find_reachable_categories ===")
    profile  = SAMPLE_PROFILES["home_decorator"]
    eligible = list(PRODUCT_CATEGORIES)
    result   = find_reachable_categories(profile, eligible)
    print(f"  Result: {result}")
    check("Returns a list", isinstance(result, list))
    check("Result is subset of eligible", all(c in eligible for c in result))
    check("Home Decor present", "Home Decor" in result)
    check("Fashion & Accessories pruned (sim=0.046 < 0.05)", "Fashion & Accessories" not in result)
    check("Returns 7 categories", len(result) == 7, f"got {len(result)}")
    check("Home Decor is first", result[0] == "Home Decor")
    # hop-1 (sim>=0.05): Food&Conf, Stationery, Garden, Kitchen, Toys
    # hop-2: Seasonal & Gifts (below 0.05 directly, reached via Stationery/Toys)
    hop1 = {"Food & Confectionery","Stationery & Craft","Garden & Outdoor","Kitchen & Dining","Toys & Games"}
    hop2 = {"Seasonal & Gifts"}
    if all(c in result for c in hop1) and all(c in result for c in hop2):
        p1 = [result.index(c) for c in hop1]
        p2 = [result.index(c) for c in hop2]
        check("BFS ordering: hop-1 before hop-2", max(p1) < min(p2),
              f"hop1={sorted(p1)} hop2={sorted(p2)}")
    else:
        print(f"  [{SKIP}] BFS ordering -- expected categories not all present")

def test_no_history():
    print("\n=== cold-start guard (no favourite_category) ===")
    eligible = ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]
    result   = find_reachable_categories(SAMPLE_PROFILES["new_customer"], eligible)
    print(f"  Result: {result}")
    check("Returns eligible unchanged", result == eligible)

def test_subset_eligible():
    print("\n=== respects eligible subset ===")
    eligible = ["Home Decor", "Kitchen & Dining", "Fashion & Accessories"]
    result   = find_reachable_categories(SAMPLE_PROFILES["home_decorator"], eligible)
    print(f"  Result: {result}")
    check("Subset of eligible", all(c in eligible for c in result))
    check("Fashion still pruned", "Fashion & Accessories" not in result)
    check("Home Decor and Kitchen present", "Home Decor" in result and "Kitchen & Dining" in result)

def test_popular():
    print("\n=== find_popular_categories ===")
    eligible = ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]
    result   = find_popular_categories(eligible, price_range="Low", top_n=3)
    print(f"  Result: {result}")
    check("Returns a list", isinstance(result, list))
    check("Up to 3 items", len(result) <= 3)
    check("Each item is (str, int)", all(isinstance(t,tuple) and isinstance(t[0],str) and isinstance(t[1],int) for t in result))
    check("Scores are int not float", all(type(t[1]) is int for t in result))
    check("Categories from eligible", all(t[0] in eligible for t in result))
    if len(result) > 1:
        check("Sorted descending", all(result[i][1] >= result[i+1][1] for i in range(len(result)-1)))

def test_popular_empty():
    print("\n=== empty eligible ===")
    result = find_popular_categories([], price_range="Low")
    print(f"  Result: {result}")
    check("Returns empty list", result == [])

def test_popular_top_n():
    print("\n=== top_n cap ===")
    result = find_popular_categories(list(PRODUCT_CATEGORIES), price_range="Mid-Low", top_n=2)
    print(f"  Result: {result}")
    check("Returns exactly 2", len(result) == 2)

if __name__ == "__main__":
    print("=" * 56)
    print("  Search Module -- Test Suite")
    print("=" * 56)
    try:
        test_reachable()
        test_no_history()
        test_subset_eligible()
        test_popular()
        test_popular_empty()
        test_popular_top_n()
        print("\nAll done.\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
