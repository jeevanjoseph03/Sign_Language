import os
import numpy as np
from pathlib import Path

DATA_PATH = Path("Sign_Language_Data")
ACTIONS = ["Hello", "How are you", "I need help", "Thank you", "Goodbye"]
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

print("=" * 60)
print("DATA COLLECTION VERIFICATION")
print("=" * 60)

total_files = 0
total_size_mb = 0

for action in ACTIONS:
    action_path = DATA_PATH / action
    if not action_path.exists():
        print(f"\n❌ {action}: NOT FOUND")
        continue
    
    action_files = 0
    action_size = 0
    
    for sequence in range(NO_SEQUENCES):
        seq_path = action_path / str(sequence)
        if seq_path.exists():
            npy_files = list(seq_path.glob("*.npy"))
            action_files += len(npy_files)
            
            for npy_file in npy_files:
                file_size = os.path.getsize(npy_file)
                action_size += file_size
                
                # Verify file integrity
                try:
                    data = np.load(npy_file)
                    if data.shape != (258,):
                        print(f"  ⚠️ {npy_file.name}: Wrong shape {data.shape}, expected (258,)")
                except Exception as e:
                    print(f"  ❌ {npy_file.name}: Error reading - {e}")
    
    expected_files = NO_SEQUENCES * SEQUENCE_LENGTH
    total_files += action_files
    total_size_mb += action_size / (1024 * 1024)
    
    status = "✅" if action_files == expected_files else "⚠️"
    print(f"\n{status} {action}")
    print(f"   Files: {action_files}/{expected_files}")
    print(f"   Size: {action_size / (1024 * 1024):.2f} MB")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
expected_total = len(ACTIONS) * NO_SEQUENCES * SEQUENCE_LENGTH
print(f"Total files collected: {total_files}/{expected_total}")
print(f"Total size: {total_size_mb:.2f} MB")
print(f"Status: {'✅ COMPLETE' if total_files == expected_total else '⚠️ INCOMPLETE'}")
print("=" * 60)
