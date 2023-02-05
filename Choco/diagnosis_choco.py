import subprocess


def get_siemens_diagnosis(variable_order_file_path=None):
    try:
        if variable_order_file_path:
            result = subprocess.run(["java", "-jar",
                                     r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\siemens_diagnosis.jar",
                                     r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs\conf_0.xml",
                                     variable_order_file_path],
                                    capture_output=True, text=True, timeout=400)
        else:
            result = subprocess.run(["java", "-jar",
                                     r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\siemens_diagnosis.jar",
                                     r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs\conf_0.xml"],
                                    capture_output=True, text=True, timeout=400)

    except:
        print('Subprocess did not answer! Continue with another try...')

    return print(result)


# get_siemens_diagnosis()
# get_camera_diagnosis(test_file_path=r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\data\camera_testing_ur.csv")
                     #variable_order_file_path=r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\data\camera_testing_vo.csv")
                     #diag_file_path=r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\data\camera_diag_output")
# get_camera_diagnosis()

