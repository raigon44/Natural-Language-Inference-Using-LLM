import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from tqdm.auto import tqdm


class Model:

    def __init__(self, num_labels, modelName):
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        self.model = BertForSequenceClassification.from_pretrained(modelName, num_labels=num_labels,
                                                                   output_attention=False, output_hidden_states=False)
        self.modelName = modelName
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def fineTune(self, epochs, warmupSteps, optimizer, trainDataLoader):

        trainingSteps = epochs * len(trainDataLoader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmupSteps,
            num_training_steps=trainingSteps
        )

        self.model.to(self.device)
        progress_bar = tqdm(range(trainingSteps))

        self.model.train()
        for epoch in range(epochs):
            for batch in trainDataLoader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_input_token_type = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)
                outputs = self.model(b_input_ids, token_type_ids=b_input_token_type, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        return

    def saveModel(self):
        self.model.save_pretrained(self.modelName+'fine_tuned.model')
        return


