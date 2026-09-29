import pandas as pd
from pdmlabs.utils.dataset import Dataset
from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from pdmlabs.RunExperiment import run_experiment

from pdmlabs.method.isolation_forest import IsolationForest


def main():
    # from pdmlabs.method.lof_semi import LocalOutlierFactor

    # 1. Load your dataset
    df = pd.read_csv("data/ims.csv")
    dataset_handler = Dataset(df, datetime_column="Artificial_timestamp", train_sources=0.33, val_sources=0.33, test_sources=0.34)

    # Extract the appropriate dataset format for your task (Unsupervised, RUL, Classification, etc.)
    Train_Val_data, Train_Test_data = dataset_handler.get_semi_dataset() 

    # 2. Define your experiment flavor
    experiments = [AutoProfileSemiSupervisedPdMExperiment]
    experiment_names = ['[IPC] My TSAD Experiment']

    # 3. Define the methods to test and their hyperparameter search spaces
    methods = [
        IsolationForest, 
    # LocalOutlierFactor
    ]

    param_space_dict_per_method = [
        {
            'n_estimators': [25, 50, 100, 200], 
            'max_samples': [100, 200, 300, 400, 500, 1000], 
            'random_state': [42], 
            'max_features': [0.25, 0.5, 0.8, 1.0], 
            'bootstrap': [True, False]
        },
        # {'n_neighbors': [2, 3, 5, 10, 20]}
    ]
    method_names = [
        "IF", 
        # "LOF"
    ]

    # 4. Execute the experiment (Hyperparameter tuning + Evaluation + MLflow Logging)
    best_params = run_experiment(
        dataset=Train_Val_data, 
        methods=methods, 
        param_space_dict_per_method=param_space_dict_per_method, 
        method_names=method_names,
        experiments=experiments, 
        experiment_names=experiment_names,
        MAX_RUNS=100, 
        MAX_JOBS=8, 
        INITIAL_RANDOM=4,
        initial_profile_size=[256],
        mlflow_port=8080, # Starts an MLflow UI server locally
        optimizer='smac',
        optimization_param='VUS_VUS_PR'
    )


# MAX_JOBS > 1 makes SMAC spawn worker processes that re-import this module.
# Without this guard the module body re-executes on import and multiprocessing
# aborts with "An attempt has been made to start a new process before the
# current process has finished its bootstrapping phase".
if __name__ == "__main__":
    main()
