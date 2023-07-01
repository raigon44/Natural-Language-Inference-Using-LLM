import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


class Model:

    def __init__(self, num_labels, modelName):
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        self.model = BertForSequenceClassification.from_pretrained(modelName, num_labels=num_labels,
                                                                   output_attentions=False, output_hidden_states=False)
        self.modelName = modelName
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def fineTune(self, epochs, warmupSteps, optimizer, trainDataLoader, validationDataLoader):

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

            train_loss = train_loss / len(trainDataLoader)

            self.model.eval()

            val_loss = 0
            with torch.no_grad():
                for batch in validationDataLoader:
                    batch = [item.to(self.device) for item in batch]
                    input_ids, attention_mask, labels = batch
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()

            val_loss = val_loss / len(validationDataLoader)

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        return

    def saveModel(self, location):
        self.model.save_pretrained(location+'/'+self.modelName+'fine_tuned.model')
        return

    def evaluate(self, test_data_loader):

        pred = []
        labels = []
        self.model.to(self.device)
        self.model.eval()
        for batch in test_data_loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_input_token_type = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=b_input_token_type,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            pred.append(predictions.tolist())
            labels.append(b_labels.tolist())

        pred = [item for sublist in pred for item in sublist]
        labels = [item for sublist in labels for item in sublist]

        print("Accuracy score (test data) : " + str(accuracy_score(pred, labels)))

        return


