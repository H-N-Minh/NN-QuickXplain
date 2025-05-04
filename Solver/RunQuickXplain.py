import subprocess
import os
import glob
import shutil

def getConflict(settings):
    try:
        # change feature model file according to the feature model to be diagnosed
        # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
        jar_path = settings["PATHS"]["SOLVER_PATH"]
        fm_path = settings["PATHS"]["SOLVER_FM_PATH"]
        log_dir = settings["PATHS"]["SOLVER_LOGS_PATH"]
        solver_input_path = settings["PATHS"]["SOLVER_INPUT_PATH"]
        os.makedirs(log_dir, exist_ok=True)
        result = subprocess.run(["java", f"-Dlog.dir={log_dir}", "-jar",jar_path, fm_path, solver_input_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # move the logs to LOGS folder
        for log_file in glob.glob("*.log"):
            shutil.move(log_file, os.path.join(log_dir, log_file))
        for zip_file in glob.glob("*.zip"):
            shutil.move(zip_file, os.path.join(log_dir, zip_file))
        for tmp_file in glob.glob("*.tmp"):
            shutil.move(tmp_file, os.path.join(log_dir, tmp_file))
            
        # moving the output folder to Solver/output folder
        output_folder = settings["PATHS"]["SOLVER_OUTPUT_PATH"]
        data_folder = "data"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        shutil.move(data_folder, output_folder)

        return result
    except:
        print('Subprocess did not answer! Continue with another try...')
        return None


