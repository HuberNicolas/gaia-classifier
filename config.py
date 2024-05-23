class DataConfig:
    data_path = './data'
    training_data_filename = 'dataGaia_AB_train.csv'
    submission_data_filename = 'dataGaia_AB_unknown.csv'


class EvaluationConfig:
    logfile_path = 'evaluation/grid_search_results.log'
    best_configs_file = 'evaluation/best_configurations_per_model.log'
    heatmap_plot_path = 'evaluation/heatmap_plot.png'
    ROC_plot_path = 'evaluation/ROC_plot.png'
    pipeline_path = 'evaluation/gaia_pipeline.joblib'


class Settings:
    dev = True
    sample_size = 0.50
    stratified_k_fold_n_splits = 5
    random_state = 31011997


class ModelConfig:
    target_column = 'SpType-ELS'
    test_size = 0.2
    all_columns = ['ID', 'Unnamed: 0', 'RA_ICRS', 'DE_ICRS', 'Source', 'Plx', 'PM', 'pmRA',
                   'pmDE', 'Gmag', 'e_Gmag', 'BPmag', 'e_BPmag', 'RPmag', 'e_RPmag',
                   'GRVSmag', 'e_GRVSmag', 'BP-RP', 'BP-G', 'G-RP', 'pscol', 'Teff',
                   'Dist', 'Rad', 'Lum-Flame', 'Mass-Flame', 'Age-Flame', 'z-Flame']
    columns_to_drop = ['ID', 'Unnamed: 0', 'Source']
    columns_to_keep = ['Teff', 'GRVSmag', 'DE_ICRS']
    configs = [
        {
            'config_nr': 'evaluation/e_1_1f', 'importance': 0.031499201614265684,
            'features': ['GRVSmag']
        },
        {
            'config_nr': 'evaluation/e_2_1f', 'importance': 0.9350919650871045,
            'features': ['Teff']
        },
        {
            'config_nr': 'evaluation/e_3_5f', 'importance': 0.9890979380932652,
            'features': ['Teff', 'GRVSmag', 'DE_ICRS', 'RA_ICRS', 'Dist']
        },
        {
            'config_nr': 'evaluation/e_4_25f', 'importance': 0.9999999999999999,
            'features': [
                'RA_ICRS', 'DE_ICRS', 'Plx', 'PM', 'pmRA', 'pmDE', 'Gmag',
                'e_Gmag', 'BPmag', 'e_BPmag', 'RPmag', 'e_RPmag', 'GRVSmag', 'e_GRVSmag',
                'BP-RP', 'BP-G', 'G-RP', 'pscol', 'Teff', 'Dist', 'Rad',
                'Lum-Flame', 'Mass-Flame', 'Age-Flame', 'z-Flame'
            ]
        },
    ]
class ImportanceConfig:
    importance_thresholds = [1.0, 0.99, 0.98, 0.96, 0.95, 0.90, 0.80, 0.50, 0.20]
    importance_plot_path = './importance/importance.png'
    importance_accumulated_path = './importance/importance_acc.png'