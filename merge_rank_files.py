"""Merge rank files that were not merged during distributed inference."""
import os
import json
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_inference_results")

# Find all rank file patterns
rank_files = glob.glob(os.path.join(OUTPUT_DIR, "*_rank*.json"))
experiments = set()

for f in rank_files:
    # Extract experiment name (remove _rankN.json)
    base = os.path.basename(f)
    exp_name = base.rsplit("_rank", 1)[0]
    experiments.add(exp_name)

print(f"Found {len(experiments)} experiments with unmerged rank files:")
for exp in sorted(experiments):
    print(f"  - {exp}")

print("\n" + "="*60)

for exp_name in sorted(experiments):
    # Find all rank files for this experiment
    pattern = os.path.join(OUTPUT_DIR, f"{exp_name}_rank*.json")
    files = sorted(glob.glob(pattern))
    
    print(f"\nProcessing: {exp_name}")
    print(f"  Found {len(files)} rank files")
    
    # Merge all results
    all_results = {}
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                all_results.update(data)
                print(f"    {os.path.basename(f)}: {len(data)} items")
        except Exception as e:
            print(f"    ERROR reading {f}: {e}")
    
    print(f"  Total merged items: {len(all_results)}")
    
    # Save merged file
    output_path = os.path.join(OUTPUT_DIR, f"{exp_name}.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")
    
    # Check if it's MCQ or open-ended and calculate metrics
    sample = next(iter(all_results.values()), {})
    if "correct" in sample:
        total_correct = sum(1 for v in all_results.values() if v.get("correct"))
        total_count = len([v for v in all_results.values() if "correct" in v])
        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"  Accuracy: {total_correct}/{total_count} = {accuracy:.4f}")
    elif "score" in sample:
        scores = [v.get("score", 0) for v in all_results.values() if "score" in v]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"  Avg Score: {avg_score:.2f} over {len(scores)} samples")
    
    # Delete rank files after successful merge
    for f in files:
        os.remove(f)
        print(f"  Deleted: {os.path.basename(f)}")

print("\n" + "="*60)
print("Done!")
