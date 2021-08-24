from dotenv import find_dotenv, load_dotenv
import mlflow
import os
import shutil
load_dotenv(find_dotenv())


if __name__ == '__main__':
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    exp_id = mlflow.get_experiment_by_name('Cust 1.5 year value').experiment_id

    # Find df of all mflow runs
    df_mlflow = mlflow.search_runs()
    df_mlflow.sort_values(by='metrics.F1_Binary', ascending=False, inplace=True)
    dir_mlflow = os.environ.get('MLFLOW_TRACKING_URI')+'/'+exp_id+'/'

    # Remove the runs with the lowest F1_Binary
    if len(df_mlflow) > 5:
        for i in range(5, len(df_mlflow)):
            rm_dir_mlflow = dir_mlflow + df_mlflow.loc[i,'run_id']
            shutil.rmtree(rm_dir_mlflow, ignore_errors=True)
