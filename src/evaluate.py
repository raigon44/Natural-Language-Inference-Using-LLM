import data_utils
import pandas as pd
from model import Model
import configparser
from transformers import BertModel


def main(dataset_name, pre_trained_model):

    train, validation, test = data_utils.load_data(dataset_name)
    test_frame = pd.DataFrame(test.to_pandas())

    model_obj = Model(3, pre_trained_model)

    config = configparser.ConfigParser()
    config.read('config.ini')
    batch_size = config.getint('Hyperparameter', 'batch_size')
    save_location = config.get('FilePaths', 'SAVED_MODEL_LOCATION')

    test_data_loader = data_utils.getDataLoader(test_frame, model_obj.tokenizer, batch_size)

    model_obj.model = BertModel.from_pretrained(save_location)

    model_obj.evaluate(test_data_loader)


if __name__ == '__main__':
    main('snli', 'bert-base-uncased')

