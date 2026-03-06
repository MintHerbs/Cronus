"""
Orchestrator_dataset.py
Master pipeline runner for the Lipstick Recommendation Dataset Generator
Executes all 5 modules sequentially and verifies outputs
"""

import subprocess
import sys
import time
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODULE_DIR = Path("module2")
OUTPUT_DIR = Path("output")

PIPELINE = [
    MODULE_DIR / "SkinToneGenerator.py",
    MODULE_DIR / "SkinTextureGenerator.py",
    MODULE_DIR / "ContrastCalculator.py",
    MODULE_DIR / "ShadeRangeEngine.py",
    MODULE_DIR / "DatasetAssembler.py",
]

EXPECTED_OUTPUTS = [
    OUTPUT_DIR / "synthetic_skin_tones.csv",
    OUTPUT_DIR / "skin_profiles_with_texture.csv",
    OUTPUT_DIR / "skin_profiles_with_contrast.csv",
    OUTPUT_DIR / "skin_profiles_with_shades.csv",
    OUTPUT_DIR / "final_skin_tone_dataset.csv",
]


def ensure_dirs():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PASS Output directory ready: {OUTPUT_DIR}")


def run_pipeline():
    """Execute all modules sequentially and collect timing results"""
    results = []
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION")
    print("=" * 70)
    
    for i, script in enumerate(PIPELINE, 1):
        name = script.name
        print(f"\n[{i}/{len(PIPELINE)}] Running {name}...")
        print("─" * 70)
        
        if not script.exists():
            print(f"ERROR: Script not found: {script}")
            print("Aborting pipeline.")
            sys.exit(1)
        
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                check=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - t0
            status = "PASSED"
            
            # Print module output
            if proc.stdout:
                print(proc.stdout)
            
            print(f"PASS {name} completed in {elapsed:.2f}s")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            status = "FAILED"
            
            print(f"FAIL {name} FAILED (exit code {e.returncode})")
            if e.stdout:
                print("\nStdout:")
                print(e.stdout)
            if e.stderr:
                print("\nStderr:")
                print(e.stderr)
            
            print(f"\nAborting pipeline after {elapsed:.2f}s")
            sys.exit(1)
        
        results.append({
            "module": name,
            "status": status,
            "time_s": round(elapsed, 2)
        })
    
    return results


def ensure_final_dataset():
    """Ensure the final dataset file exists"""
    final_dataset_path = OUTPUT_DIR / "final_skin_tone_dataset.csv"
    
    if not final_dataset_path.exists():
        print("Creating final dataset file...")
        try:
            # Load the shades file and create final dataset
            import pandas as pd
            df = pd.read_csv(OUTPUT_DIR / "skin_profiles_with_shades.csv")
            
            # Prepare final schema
            column_mapping = {
                'L': 'skin_L',
                'a': 'skin_a', 
                'b': 'skin_b',
                'normal_pct': 'normal_pct',
                'oily_pct': 'oily_pct',
                'dry_pct': 'dry_pct',
                'Contrast_Level': 'contrast_level',
                'primary_group': 'primary_group',
                'sub_group': 'sub_group',
                'shade_L_min': 'shade_L_min',
                'shade_L_max': 'shade_L_max',
                'shade_a_min': 'shade_a_min',
                'shade_a_max': 'shade_a_max',
                'shade_b_min': 'shade_b_min',
                'shade_b_max': 'shade_b_max'
            }
            
            # Select and rename columns
            final_df = df[list(column_mapping.keys())].copy()
            final_df = final_df.rename(columns=column_mapping)
            final_df['contrast_level'] = final_df['contrast_level'].str.lower()
            final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save final dataset
            final_df.to_csv(final_dataset_path, index=False)
            print(f"Final dataset created: {final_dataset_path}")
            
        except Exception as e:
            print(f"Error creating final dataset: {e}")
    else:
        print(f"Final dataset already exists: {final_dataset_path}")


def verify_outputs():
    """Verify that all expected output files exist"""
    print("\n" + "=" * 70)
    print("FILE VERIFICATION")
    print("=" * 70)
    
    all_found = True
    
    for filepath in EXPECTED_OUTPUTS:
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"FOUND  {filepath.name} ({size_kb:.1f} KB)")
        else:
            print(f"MISSING {filepath.name}")
            all_found = False
    
    return all_found


def preview_dataset():
    """Load and preview the final dataset"""
    final_dataset = OUTPUT_DIR / "final_skin_tone_dataset.csv"
    
    if not final_dataset.exists():
        print("\nFinal dataset not found - cannot preview")
        return
    
    print("\n" + "=" * 70)
    print("FINAL DATASET PREVIEW")
    print("=" * 70)
    
    df = pd.read_csv(final_dataset)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def prompt_for_visualizations():
    """Prompt user if they want to generate dataset health visualizations"""
    print("\n" + "=" * 70)
    print("DATASET VISUALIZATION OPTIONS")
    print("=" * 70)
    
    while True:
        response = input("\nGenerate dataset health visualizations? (Y/N): ").strip().upper()
        
        if response == 'Y':
            print("\nGenerating comprehensive dataset visualizations...")
            print("This will create 5 detailed plots for dataset validation.")
            
            try:
                # Run the visualization script
                result = subprocess.run(
                    [sys.executable, "module2/generate_visualizations.py"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd="."  # Ensure we're running from the root directory
                )
                
                if result.returncode == 0:
                    print("\n" + "=" * 70)
                    print("VISUALIZATION GENERATION COMPLETE")
                    print("=" * 70)
                    print("Dataset generated: output/final_skin_tone_dataset.csv")
                    print("Visualizations saved to: module2/output/")
                    print("\nGenerated plots:")
                    print("  • plot1_pairplot.png/svg          — feature relationships by shade group")
                    print("  • plot2_shade_bands.png/svg       — shade L* range bands per sub group")
                    print("  • plot3_skin_tone_scatter.png/svg — skin tone space colored by group")
                    print("  • plot4_class_balance.png/svg     — record count per sub group")
                    print("  • plot5_texture_composition.png/svg — texture distribution validation")
                    print("=" * 70)
                else:
                    print("ERROR: Visualization generation failed")
                    if result.stderr:
                        print(result.stderr)
                        
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to generate visualizations (exit code {e.returncode})")
                if e.stderr:
                    print(e.stderr)
            except Exception as e:
                print(f"ERROR: Unexpected error during visualization generation: {e}")
            
            break
            
        elif response == 'N':
            print("\n" + "=" * 70)
            print("DATASET GENERATION COMPLETE")
            print("=" * 70)
            print("Dataset generated inside output folder at root as final_skin_tone_dataset.csv")
            print("=" * 70)
            break
            
        else:
            print("Please enter Y or N")


def print_final_summary(results, all_files_found):
    """Print final summary table"""
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Module':<35} {'Status':<10} {'Time (s)':>10}")
    print("─" * 70)
    
    total_time = 0
    for r in results:
        print(f"{r['module']:<35} {r['status']:<10} {r['time_s']:>10.2f}")
        total_time += r['time_s']
    
    print("─" * 70)
    print(f"{'TOTAL':<35} {'':<10} {total_time:>10.2f}")
    
    print("\n" + "=" * 70)
    
    if all_files_found and all(r['status'] == 'PASSED' for r in results):
        print("PASS PIPELINE COMPLETED SUCCESSFULLY")
        print("PASS All modules passed")
        print("PASS All output files generated")
    else:
        print("WARN PIPELINE COMPLETED WITH ISSUES")
        if not all(r['status'] == 'PASSED' for r in results):
            print("WARN Some modules failed")
        if not all_files_found:
            print("WARN Some output files missing")
    
    print("=" * 70)


def main():
    """Main execution function"""
    print("=" * 70)
    print("LIPSTICK RECOMMENDATION DATASET — ORCHESTRATOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Ensure directories exist
    ensure_dirs()
    
    # Step 2: Run pipeline
    results = run_pipeline()
    
    # Step 3: Ensure final dataset exists
    ensure_final_dataset()
    
    # Step 4: Verify outputs
    all_files_found = verify_outputs()
    
    # Step 5: Preview final dataset
    preview_dataset()
    
    # Step 6: Print summary
    print_final_summary(results, all_files_found)
    
    # Step 7: Prompt for visualizations
    prompt_for_visualizations()
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
