from model import Model
import data_utils
import logging_config


def main(pre_trained_model, dataset_name):

    logger = logging_config.config_logging()
    if dataset_name == 'snli' or dataset_name == 'multi_nli':
        num_labels = 3
    else:
        logger.error('Dataset Unknown!!')

    train, validation, test = data_utils.load_data(dataset_name)

    model_obj = Model(num_labels, pre_trained_model)
