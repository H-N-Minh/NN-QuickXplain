import json
import os
from Trainer import startTraining
from Tester import startTesting
import Utils as Utils


def main():
    print("\n\n================== DECISION TREE ===========================")
    settings = Utils.loadSettings()
    error_count = 0

    print("\n\n################## CLEANING PHASE ##########################")
    if not settings['CLEAR']['SKIP']:
        Utils.startClearing(settings)
            
    print("\n\n################## TRAINING PHASE ##########################")
    error_list = []         # store error messages during training
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        error_list = startTraining(settings)
    else:
        print("\n\n<Training phase skipped (as per settings.yaml file)>")

    print("\n\n################## Testing PHASE ########################")
    if not settings['WORKFLOW']['Test']['SKIP']:
        startTesting(settings)
    else:
        print("\n<Testing phase skipped (as per settings.yaml file)>")
    
    # Print the error messages if any
    if error_list:
        print(f"\n✖✖✖ Process got {len(error_list)} error(s) ✖✖✖\n")
        for config, error in error_list:
            print(f"✖ Trial number {config} has error message: '{error}'")
        print("\nCheck the logs above for details.")
    else:
        print("\n===Process completed successfully!===\n")
        
    


if __name__ == "__main__":
    main()

