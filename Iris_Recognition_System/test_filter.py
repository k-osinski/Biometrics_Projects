#!/usr/bin/env python3
import sys
sys.path.insert(0, '/sessions/loving-great-ride/mnt/Biometrics_Projects/Iris_Recognition_System')

from PIL import Image
import numpy as np
from src.pipeline import run_iris_pipeline
from src.core.morphology import label_components

print("=" * 60)
print("TEST 1: Real iris image with pupil_keep_largest parameter")
print("=" * 60)

try:
    img = Image.open('MMU-Iris-Database/1/left/aeval1.bmp').convert('L')
    print(f"Loaded image: {img.size}, mode: {img.mode}")
    
    # Test with filter ON
    result_on = run_iris_pipeline(img, pupil_keep_largest=True)
    mask_on = result_on.segmentation.pupil.binary_mask
    components_on = label_components(mask_on)
    cx_on, cy_on = result_on.segmentation.pupil.center
    r_on = result_on.segmentation.pupil.radius
    print(f"\nWith pupil_keep_largest=True:")
    print(f"  Center: ({cx_on:.2f}, {cy_on:.2f}), Radius: {r_on:.2f}")
    print(f"  Connected components in mask: {components_on}")
    
    # Test with filter OFF
    result_off = run_iris_pipeline(img, pupil_keep_largest=False)
    mask_off = result_off.segmentation.pupil.binary_mask
    components_off = label_components(mask_off)
    cx_off, cy_off = result_off.segmentation.pupil.center
    r_off = result_off.segmentation.pupil.radius
    print(f"\nWith pupil_keep_largest=False:")
    print(f"  Center: ({cx_off:.2f}, {cy_off:.2f}), Radius: {r_off:.2f}")
    print(f"  Connected components in mask: {components_off}")
    
    # Verify filter effect
    if components_on <= 1:
        print("\n✓ TEST 1 PASSED: Filter produces ≤1 component")
    else:
        print(f"\n✗ TEST 1 FAILED: Expected ≤1 component, got {components_on}")
        
except Exception as e:
    print(f"✗ TEST 1 FAILED with exception:\n{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Synthetic two-blob image")
print("=" * 60)

try:
    from src.core.morphology import keep_largest_component
    
    # Create synthetic image with two blobs
    synthetic = np.zeros((100, 100), dtype=np.uint8)
    synthetic[0:30, 0:30] = 255      # 30x30 blob
    synthetic[70:80, 80:90] = 255    # 10x10 blob
    
    print(f"Original image: 2 blobs (30x30 and 10x10)")
    print(f"  Blob 1 size: 30*30*255 = {30*30*255}")
    print(f"  Blob 2 size: 10*10*255 = {10*10*255}")
    
    # Label components in original
    comps_orig = label_components(synthetic)
    print(f"  label_components(original): {comps_orig} components")
    
    # Apply filter
    filtered = keep_largest_component(synthetic)
    surviving_sum = np.sum(filtered)
    expected_sum = 30 * 30 * 255
    
    print(f"\nAfter keep_largest_component:")
    print(f"  Sum of pixels: {surviving_sum}")
    print(f"  Expected (big blob only): {expected_sum}")
    
    if surviving_sum == expected_sum:
        print("\n✓ TEST 2 PASSED: Filter keeps only the large blob")
    else:
        print(f"\n✗ TEST 2 FAILED: Expected {expected_sum}, got {surviving_sum}")
        
except Exception as e:
    print(f"✗ TEST 2 FAILED with exception:\n{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
