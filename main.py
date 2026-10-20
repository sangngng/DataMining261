"""
CO3029 Data Mining - Root Runner Proxy
This wrapper allows running the pipeline directly from the root repository directory
by delegating to source-code/main.py.
"""
import os
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    source_code_dir = os.path.join(project_root, "source-code")
    sys.path.insert(0, source_code_dir)
    os.chdir(source_code_dir)
    
    from main import main
    main()
