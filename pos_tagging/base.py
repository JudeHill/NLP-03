from datasets import Dataset


class BaseUnsupervisedClassifier:
    def train(self, inputs: Dataset):
        raise NotImplementedError

    def inference(self, input_ids):
        raise NotImplementedError
