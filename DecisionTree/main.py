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
    
    # Get the original model path and replace 'busybox' with 'old_busybox'
    old_model_path = settings['PATHS']['MODEL_PATH'].replace('busybox', 'old_busybox')

    # List all json files in the old_model_path that match the pattern
    metrics_list = []
    for filename in os.listdir(old_model_path):
        if filename.endswith("_metrics.json"):
            with open(os.path.join(old_model_path, filename), 'r') as f:
                metrics_list.append(json.load(f))
    assert len(metrics_list) == 6, "loaded wrong"

    assert len(settings['WORKFLOW']['TRAIN']['configurations']['test_sizes']) > 0, "test_sizes should not be empty"
    assert len(settings['WORKFLOW']['TRAIN']['configurations']['max_depths']) > 0, "max_depths should not be empty"
    assert len(settings['WORKFLOW']['TRAIN']['configurations']['estimator_types']) > 0, "estimator_types should not be empty"
    assert len(settings['WORKFLOW']['TRAIN']['configurations']['multi_output_types']) > 0, "multi_output_types should not be empty"
    assert len(settings['WORKFLOW']['TRAIN']['configurations']['use_pca_options']) > 0, "use_pca_options should not be empty"
    assert len(settings['WORKFLOW']['TRAIN']['configurations']['class_weight_options']) > 0, "class_weight_options should not be empty"
    for jsn in metrics_list:
        settings['WORKFLOW']['TRAIN']['configurations']['test_sizes'] = [jsn['config']['test_size']]
        assert (
            len(settings['WORKFLOW']['TRAIN']['configurations']['test_sizes']) == 1 and
            isinstance(settings['WORKFLOW']['TRAIN']['configurations']['test_sizes'][0], float)
        ), "test_sizes should be a list containing exactly one float"

        settings['WORKFLOW']['TRAIN']['configurations']['max_depths'] = [jsn['config']['max_depth']]
        settings['WORKFLOW']['TRAIN']['configurations']['estimator_types'] = [jsn['config']['estimator_type']]
        settings['WORKFLOW']['TRAIN']['configurations']['multi_output_types'] = [jsn['config']['multi_output_type']]
        settings['WORKFLOW']['TRAIN']['configurations']['use_pca_options'] = [jsn['config']['use_pca']]
        settings['WORKFLOW']['TRAIN']['configurations']['class_weight_options'] = [jsn['config']['class_weight']]
        print(f"training the following config: test_size={jsn['config']['test_size']}, max_depth={jsn['config']['max_depth']}, "
              f"estimator_type={jsn['config']['estimator_type']}, multi_output_type={jsn['config']['multi_output_type']}, "
              f"use_pca={jsn['config']['use_pca']}, class_weight={jsn['config']['class_weight']}")
        error_count += startTraining(settings)
        assert error_count == 0, "Training failed with errors"


            
    # print("\n\n################## TRAINING PHASE ##########################")
    # if not settings['WORKFLOW']['TRAIN']['SKIP']:
    #     error_count = startTraining(settings)
    # else:
    #     print("\n\n<Training phase skipped (as per settings.yaml file)>")

    # print("\n\n################## VALIDATION PHASE ########################")
    # if not settings['WORKFLOW']['VALIDATE']['SKIP']:
    #     startTesting(settings)
    # else:
    #     print("\n<Validation phase skipped (as per settings.yaml file)>")
    
    if error_count > 0:
        print(f"Process completed with {error_count} error(s). Please check the logs above for details.")
    else:
        print("\n===Process completed successfully!===\n")
        
    


if __name__ == "__main__":
    main()

