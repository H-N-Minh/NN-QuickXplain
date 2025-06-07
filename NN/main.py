from Trainer import startTraining
# from Tester import startTesting
import Utils as Utils


def main():
    print("\n==================== NEURAL NETWORK ===========================")
    settings = Utils.loadSettings()

    print("\n\n################## CLEANING PHASE ##########################")
    if not settings['CLEAR']['SKIP']:
        Utils.startClearing(settings)
            
    print("\n\n################## TRAINING PHASE ##########################")
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        error_list = startTraining(settings)
    else:
        print("\n\n<Training phase skipped (as per settings.yaml file)>")

    if error_list:
        print(f"\n===Process completed with {len(error_list)} error(s)===\n")
        for config, error in error_list:
            print(f"Error with config: {config}\n==> Error: {error}")
        print("Check the logs above for details.")
    else:
        print("\n===Process completed successfully!===\n")




if __name__ == "__main__":
    main()