"""
notebooks/test_rules.py
"""
from src.constants import SAMPLE_PROFILES
from src.rules_engine import apply_rules

def run_verification_testsuite():
    print("=" * 70)
    print("CIC6314 RULES ENGINE TEST MATRIX ENGINE RUN")
    print("=" * 70)
    
    for name, profile in SAMPLE_PROFILES.items():
        result = apply_rules(profile)
        print(f"\n[Profile ID]: {name}")
        print(f" -> Segment : {profile.get('customer_segment')}")
        print(f" -> Spend   : {profile.get('price_range')}")
        print(f" -> Eligible: {result}")

if __name__ == "__main__":
    run_verification_testsuite()