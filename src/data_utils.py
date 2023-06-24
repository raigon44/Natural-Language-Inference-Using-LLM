from datasets import load_dataset


def load_data(dataset_name):
    """Loads the dataset from Hugging Face dataset library and returns the train, test and validation splits"""
    data = load_dataset(dataset_name)
    return data['train'], data['validation'], data['test']


