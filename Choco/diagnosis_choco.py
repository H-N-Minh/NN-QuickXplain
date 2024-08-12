import subprocess


def get_linux_diagnosis(configuration_file_path):
    try:
        # change feature model file according to the feature model to be diagnosed
        # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
        result = subprocess.run(["java", "-jar",
                                 r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnosis.jar",
                                 r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                                 configuration_file_path])
    except:
        print('Subprocess did not answer! Continue with another try...')

    return print(result)

