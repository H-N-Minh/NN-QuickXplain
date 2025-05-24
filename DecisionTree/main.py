import sys
import os
import yaml

from Trainer import startTraining
from Tester import startTesting


def loadSettings():
    """Load settings from YAML file."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Construct the absolute path to the settings.yaml file
        settings_path = os.path.join(root_dir, 'settings.yaml')

        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
    except FileNotFoundError:
        print("Settings file not found. Please make sure the settings.yaml file is in the correct directory.")
        sys.exit(1)
    
    for key in settings['PATHS']:
        settings['PATHS'][key] = os.path.join(root_dir, settings['PATHS'][key])
    return settings



def main():
    settings = loadSettings()
    
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

