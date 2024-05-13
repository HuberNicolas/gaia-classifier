class DataConfig:
    data_path = './data'
    training_data_filename = 'dataGaia_AB_train.csv'
    submission_data_filename = 'dataGaia_AB_unknown.csv'


class Settings:
    dev = False
    sample_size = 0.1
    stratified_k_fold_n_splits = 10
    random_state = 31011997


class ModelConfig:
    target_column = 'SpType-ELS'
    test_size = 0.2
    columns_to_drop = ['Unnamed: 0', 'Source']
