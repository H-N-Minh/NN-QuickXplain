from Trainer import startTraining
from Tester import startTesting
import Utils as Utils


def main():
    settings = Utils.loadSettings()

    print("\n\n################## CLEANING PHASE ##########################")
    if not settings['CLEAR']['SKIP']:
        Utils.startClearing(settings)
            
    print("\n\n################## TRAINING PHASE ##########################")
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        startTraining(settings)
    else:
        print("\n\n<Training phase skipped (as per settings.yaml file)>")

    print("\n\n################## VALIDATION PHASE ########################")
    if not settings['WORKFLOW']['VALIDATE']['SKIP']:
        startTesting(settings)
    else:
        print("\n<Validation phase skipped (as per settings.yaml file)>")

    print("\n===Process completed successfully!===\n")
        
    


if __name__ == "__main__":
    main()

