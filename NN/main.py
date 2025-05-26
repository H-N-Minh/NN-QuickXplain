from Trainer import startTraining
from Tester import startTesting
import Utils as Utils


def main():
    print("\n==================== NEURAL NETWORK ===========================")
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





    # # Import
    # settings = DataHandling.importSettings()
    # features_dataframe, labels_dataframe = DataHandling.importTrainingData(settings)
    
    # # Preprocess
    # features_dataframe, labels_dataframe = DataHandling.preprocessTrainingData(features_dataframe, labels_dataframe)

    # # Create and train model
    # constraint_size = features_dataframe.shape[1] # Number of features/labels
    # NN_model = Model.ConflictNN(constraints_size= constraint_size, settings= settings, 
    #                             constraint_name_list= features_dataframe.columns.tolist())
    # NN_model.prepareData(features_dataframe, labels_dataframe)
    # NN_model.train()


if __name__ == "__main__":
    main()