import subprocess


def get_linux_diagnosis(configuration_file_path):
    try:
        result = subprocess.run(["java", "-jar",
                                 r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnosis.jar",
                                 r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\linux-2.6.33.3.xml",
                                 configuration_file_path])
    except:
        print('Subprocess did not answer! Continue with another try...')

    return print(result)

