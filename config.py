class DataConfig:
    data_path = './data'
    training_data_filename = 'dataGaia_AB_train.csv'
    submission_data_filename = 'dataGaia_AB_unknown.csv'


class EvaluationConfig:
    logfile_path = './evaluation/grid_search_results.log'
    best_configs_file = './evaluation/best_configurations_per_model.log'
    heatmap_plot_path = './evaluation/heatmap_plot.png'
    ROC_plot_path = './evaluation/ROC_plot.png'
    pipeline_path = './evaluation/gaia_pipeline.joblib'


class Settings:
    dev = False
    sample_size = 0.001
    stratified_k_fold_n_splits = 10
    random_state = 31011997


class ModelConfig:
    target_column = 'SpType-ELS'
    test_size = 0.2
    columns_to_drop = ['ID', 'Unnamed: 0', 'Source']
