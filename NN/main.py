from Trainer import startTraining
# from Tester import startTesting
import Utils as Utils


def main():
    print("\n==================== NEURAL NETWORK ===========================")
    settings = Utils.loadSettings()
    error_count = 0

    print("\n\n################## CLEANING PHASE ##########################")
    if not settings['CLEAR']['SKIP']:
        Utils.startClearing(settings)
            
    print("\n\n################## TRAINING PHASE ##########################")
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        error_count = startTraining(settings)
    else:
        print("\n\n<Training phase skipped (as per settings.yaml file)>")

    if error_count > 0:
        print(f"Process completed with {error_count} error(s). Please check the logs above for details.")
    else:
        print("\n===Process completed successfully!===\n")




if __name__ == "__main__":
    main()