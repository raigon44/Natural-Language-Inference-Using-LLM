from model import Model
import data_utils
import logging_config
import configparser
from transformers import AdamW


def main(pre_trained_model, dataset_name):

    logger = logging_config.config_logging()
    if dataset_name == 'snli' or dataset_name == 'multi_nli':
        num_labels = 3
    else:
        logger.error('Dataset Unknown!!')

    train, validation, test = data_utils.load_data(dataset_name)

    config = configparser.ConfigParser()
    config.read('config.ini')

    learning_rate = config.getfloat('Hyperparameter', 'learning_rate')
    batch_size = config.getint('Hyperparameter', 'batch_size')
    num_epochs = config.getint('Hyperparameter', 'num_epochs')
    num_warmup = config.getint('Hyperparameter', 'num_warmup_steps')
    optimizer = config.get('Hyperparameter', 'optimizer')

    save_location = config.get('FilePaths', 'SAVED_MODEL_LOCATION')

    model_obj = Model(num_labels, pre_trained_model)

    if optimizer == 'AdamW':
        optimizer = AdamW(model_obj.model.parameters(), lr=learning_rate)
    else:
        logger.error('Unknown Optimizer !!')

    train_data_loader = data_utils.getDataLoader(train, model_obj.tokenizer, batch_size)

    logger.info('Fine Tuning the model...')

    model_obj.fineTune(num_epochs, num_warmup, optimizer, train_data_loader)

    logger.info('Trying to save the model....')

    model_obj.saveModel(save_location)

    logger.info('Model Saved!! at'+save_location)



