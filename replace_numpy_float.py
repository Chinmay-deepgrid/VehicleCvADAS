#!/usr/bin/env python3
import os
import re

def replace_numpy_float(directory):
    """
    Recursively replace NumPy float-related deprecated usages
    """
    replacements = [
        (r'(?<!import\s)(?<!#.*)\bnp\.float\b', 'np.float64'),
        (r'dtype=np\.float\b', 'dtype=np.float64'),
        (r'dtype=float\b', 'dtype=np.float64')
    ]
    
    modified_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    original_content = content
                    for pattern, replacement in replacements:
                        content = re.sub(pattern, replacement, content)
                    
                    if original_content != content:
                        with open(filepath, 'w') as f:
                            f.write(content)
                        modified_files.append(filepath)
                        print(f"Updated: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return modified_files

# Use the script
if __name__ == '__main__':
    project_directory = '/home/aravind/Vehicle-CV-ADAS'
    modified = replace_numpy_float(project_directory)
    print(f"\nTotal files modified: {len(modified)}")
    if modified:
        print("Modified files:")
        for file in modified:
            print(file)