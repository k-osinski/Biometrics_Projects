from PIL import Image
import numpy as np
import sys
import traceback

# Load image
img = Image.open('MMU-Iris-Database/1/left/aeval1.bmp').convert('L')
img_array = np.array(img)

print("="*60)
print("TESTING X_I SENSITIVITY (Iris Radius)")
print("="*60)

try:
    from src.pipeline import run_iris_pipeline
    
    x_i_values = [0.7, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8]
    
    for x_i in x_i_values:
        result = run_iris_pipeline(img_array, x_i=x_i)
        iris = result.iris
        print(f"x_i={x_i:>4} → iris_radius={iris.radius:>7.2f}, method={iris.method_used:>10}, threshold={iris.threshold:>6.1f}")
    
    print("\n" + "="*60)
    print("TESTING X_P SENSITIVITY (Pupil Radius)")
    print("="*60)
    
    x_p_values = [2.5, 3.0, 3.5]
    
    for x_p in x_p_values:
        result = run_iris_pipeline(img_array, x_p=x_p)
        pupil = result.pupil
        print(f"x_p={x_p} → pupil_radius={pupil.radius:>7.2f}")
        
except Exception as e:
    print(f"ERROR:\n{traceback.format_exc()}")
    sys.exit(1)
