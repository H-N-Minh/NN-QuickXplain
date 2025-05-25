from Trainer import startTraining
from Tester import startTesting
import Utils as Utils


def main():
    settings = Utils.loadSettings()

    if not settings['CLEAR']['SKIP']:
        Utils.startClearing(settings)
            
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        startTraining(settings)
    else:
        print("\n\n<Training phase skipped (as per settings.yaml file)>")

    if not settings['WORKFLOW']['VALIDATE']['SKIP']:
        startTesting(settings)
    else:
        print("\n<Validation phase skipped (as per settings.yaml file)?")

    print("\nProcess completed successfully!\n")
        
    


if __name__ == "__main__":
    main()

