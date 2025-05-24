import subprocess
import os
import glob
import shutil

def getConflict(settings):
    try:
        # change feature model file according to the feature model to be diagnosed
        # (e.g., arcade-game.splx, busybox-1.18.0.xml, REAL-FM-4.sxfm)

        # import all the paths from the settings.yaml
        jar_path = settings["PATHS"]["SOLVER_PATH"]
        fm_path = settings["PATHS"]["TRAINDATA_FM_PATH"]
        log_dir = settings["PATHS"]["SOLVER_LOGS_PATH"]
        solver_input_path = settings["PATHS"]["SOLVER_INPUT_PATH"]
        java_exc_path = settings["PATHS"]["JAVA_PATH"]
        
        # error handling to make sure the files/folders are correct
        if not os.path.isfile(jar_path):
            raise FileNotFoundError(f"JAR file not found: {jar_path}")
        if not os.path.isfile(fm_path):
            raise FileNotFoundError(f"Feature model file not found: {fm_path}")
        if not os.path.isdir(log_dir):  # if log directory does not exist, create it
            os.makedirs(log_dir, exist_ok=True)
        if not os.path.isdir(solver_input_path) or not os.listdir(solver_input_path):
            raise FileNotFoundError(f"Input directory does not exist or is empty: {solver_input_path}")
        if java_exc_path != "java":
            java_exc_path = os.path.abspath(java_exc_path)
            if not os.path.isfile(java_exc_path):
                raise FileNotFoundError(f"Java executable not found: {java_exc_path}. Is Java installed and path is correct?")


        # running the .jar file to get the conflicts
        print("...Running QuickXplain...")
        result = subprocess.run([java_exc_path, f"-Dlog.dir={log_dir}", "-jar",jar_path, fm_path, solver_input_path], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup: move the logs to LOGS folder and output folder to Solver/output folder
        for log_file in glob.glob("*.log"):
            shutil.move(log_file, os.path.join(log_dir, log_file))
        for zip_file in glob.glob("*.zip"):
            shutil.move(zip_file, os.path.join(log_dir, zip_file))
        for tmp_file in glob.glob("*.tmp"):
            shutil.move(tmp_file, os.path.join(log_dir, tmp_file))
        output_folder = settings["PATHS"]["SOLVER_OUTPUT_PATH"]
        data_folder = "data"
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        shutil.move(data_folder, output_folder)

        return result
    except:
        assert False, "Subprocess (QuickXplain) did not answer! Make sure Java is installed and correctly pathed"


