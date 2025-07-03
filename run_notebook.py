#!/usr/bin/env python3
"""
ノートブックを実行するためのスクリプト
"""

import subprocess
import sys

def run_notebook():
    print("=== ノートブック実行開始 ===")
    
    # Jupyter notebookをPythonスクリプトに変換して実行
    cmd = [
        "jupyter", "nbconvert",
        "--to", "python",
        "--execute",
        "can_gpu_benchmark_fixed.ipynb",
        "--output", "executed_notebook.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"エラー: {result.stderr}")
            return False
        print("ノートブックが正常に実行されました")
        return True
    except Exception as e:
        print(f"実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = run_notebook()
    sys.exit(0 if success else 1)