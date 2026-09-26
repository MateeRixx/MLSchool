"""
Master pipeline runner for Entity Resolution.

Orchestrates: Blocking -> Features -> Training -> Prediction
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("Blocking", "src/01_blocking.py"),
    ("Features", "src/02_features.py"),
    ("Training", "src/03_model.py"),
    ("Prediction", "src/04_predict.py"),
]


def run_script(name: str, script: str) -> bool:
    """Run a pipeline script and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name} ({script})")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,  # Stream output directly
        text=True,
    )
    
    if result.returncode == 0:
        print(f"\n✓ {name} COMPLETED SUCCESSFULLY")
        return True
    else:
        print(f"\n✗ {name} FAILED (exit code: {result.returncode})")
        return False


def main():
    """Run full pipeline."""
    print("="*60)
    print("ENTITY RESOLUTION - FULL PIPELINE")
    print("="*60)
    
    # Check required files exist
    required = [
        "dataset/train/train_source1.tsv",
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv",
        "dataset/train/train_ground_truth.tsv",
        "dataset/test/test_source1.tsv",
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv",
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print("MISSING REQUIRED FILES:")
        for f in missing:
            print(f"  {f}")
        return
    
    print("All required data files found.")
    
    # Run each stage
    for name, script in SCRIPTS:
        if not Path(script).exists():
            print(f"SCRIPT NOT FOUND: {script}")
            return
        
        success = run_script(name, script)
        if not success:
            print(f"\nPipeline stopped at {name}")
            return
    
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Output files:")
    print("  output/matching_results.tsv")
    print("  output/candidate_pairs.tsv")
    print("  output/models/xgb_calibrated.pkl")


if __name__ == "__main__":
    main()