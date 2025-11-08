#!/usr/bin/env python3
"""
Quick script to verify all dataset files are accessible
and count the total number of samples
"""

import os

def count_wav_files(directory):
    """Count .wav files in a directory"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.wav')])
    return count

def main():
    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    
    # GITDATA
    print("\n📁 GITDATA:")
    print("-" * 70)
    
    gitdata_base = "data/gitdata"
    
    # Denoised
    denoised_path = os.path.join(gitdata_base, "denoised-speech-dataset", "Faces")
    if os.path.exists(denoised_path):
        for folder in sorted(os.listdir(denoised_path)):
            folder_path = os.path.join(denoised_path, folder)
            if os.path.isdir(folder_path):
                count = count_wav_files(folder_path)
                print(f"  ✓ denoised-speech-dataset/Faces/{folder}: {count} files")
    
    denoised_total = count_wav_files(denoised_path) if os.path.exists(denoised_path) else 0
    print(f"  SUBTOTAL (denoised): {denoised_total} files")
    
    # Original
    original_path = os.path.join(gitdata_base, "original-speech-dataset", "Faces")
    if os.path.exists(original_path):
        print()
        for folder in sorted(os.listdir(original_path)):
            folder_path = os.path.join(original_path, folder)
            if os.path.isdir(folder_path):
                count = count_wav_files(folder_path)
                print(f"  ✓ original-speech-dataset/Faces/{folder}: {count} files")
    
    original_total = count_wav_files(original_path) if os.path.exists(original_path) else 0
    print(f"  SUBTOTAL (original): {original_total} files")
    
    gitdata_total = denoised_total + original_total
    print(f"\n  📊 TOTAL GITDATA: {gitdata_total} files (all Parkinson's)")
    
    # FIGDATA
    print("\n📁 FIGDATA:")
    print("-" * 70)
    
    figdata_base = "data/figdata"
    
    hc_path = os.path.join(figdata_base, "HC_AH")
    hc_count = count_wav_files(hc_path)
    print(f"  ✓ HC_AH (Healthy Controls): {hc_count} files")
    
    pd_path = os.path.join(figdata_base, "PD_AH")
    pd_count = count_wav_files(pd_path)
    print(f"  ✓ PD_AH (Parkinson's Disease): {pd_count} files")
    
    figdata_total = hc_count + pd_count
    print(f"\n  📊 TOTAL FIGDATA: {figdata_total} files ({hc_count} healthy + {pd_count} Parkinson's)")
    
    # GRAND TOTAL
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    total = gitdata_total + figdata_total
    
    total_healthy = hc_count
    total_parkinsons = gitdata_total + pd_count
    
    print(f"  Total Healthy Controls: {total_healthy} files")
    print(f"  Total Parkinson's Patients: {total_parkinsons} files")
    print(f"  GRAND TOTAL: {total} audio files")
    print()
    print(f"  Class Distribution:")
    print(f"    Healthy: {total_healthy/total*100:.1f}%")
    print(f"    Parkinson's: {total_parkinsons/total*100:.1f}%")
    print()
    
    # Estimate after augmentation
    augmented_estimate = total * 3  # Conservative estimate with 3x augmentation
    print(f"  Estimated samples after augmentation: ~{augmented_estimate}")
    print(f"  Estimated training samples (70%): ~{int(augmented_estimate * 0.7)}")
    print(f"  Estimated validation samples (15%): ~{int(augmented_estimate * 0.15)}")
    print(f"  Estimated test samples (15%, no aug): ~{int(total * 0.15)}")
    
    print("\n" + "=" * 70)
    
    if total > 0:
        print("✅ Dataset verification complete!")
        print(f"   Ready to train with {total} audio files")
    else:
        print("⚠️  WARNING: No audio files found!")
        print("   Please check your data directory structure")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
