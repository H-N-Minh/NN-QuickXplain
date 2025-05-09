import subprocess
import os
import glob
import shutil

def get_linux_diagnosis(configuration_file_path):
    try:
        # change feature model file according to the feature model to be diagnosed
        # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
        jar_path = os.path.join("Solver", "fm_conflict.jar")
        fm_path = os.path.join("Solver", "arcade-game.splx")
        log_dir = os.path.join("Solver", "Logs")  # Change this to your desired log directory
        os.makedirs(log_dir, exist_ok=True)
        result = subprocess.run(["java", f"-Dlog.dir={log_dir}", "-jar",jar_path, fm_path, configuration_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # move the logs to LOGS folder
        for log_file in glob.glob("*.log"):
            shutil.move(log_file, os.path.join(log_dir, log_file))
        for zip_file in glob.glob("*.zip"):
            shutil.move(zip_file, os.path.join(log_dir, zip_file))
        for tmp_file in glob.glob("*.tmp"):
            shutil.move(tmp_file, os.path.join(log_dir, tmp_file))
        
        data_dir = "data"
        output_dir = os.path.join("Solver", "Output")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(data_dir):
            shutil.move(data_dir, output_dir)
        # print(result.stdout)
        return result
    except:
        print('Subprocess did not answer! Continue with another try...')
        return None


