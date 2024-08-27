import pandas as pd
import numpy as np
# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams

# import matplotlib.pyplot as plt
# from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)

df = pd.read_csv("filename1.csv")
train_df, val_df = train_test_split(df, test_size=0.2)
# train_df.shape, val_df.shape


LABEL_COLUMNS = df.columns.tolist()[3:]
# df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh")
# train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
# train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
print(len(LABEL_COLUMNS))

BERT_MODEL_NAME = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
#print(tokenizer)

# token_counts = []


MAX_TOKEN_COUNT = 500

class ToxicCommentsDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 500
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class ToxicCommentDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, tokenizer, batch_size=16, max_token_len=500):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = ToxicCommentsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = ToxicCommentsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1
        )

class ToxicCommentTagger(pl.LightningModule):

    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        #print(outputs)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            accuracy1 = accuracy(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", accuracy1, self.current_epoch)
    def validation_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            accuracy1 = accuracy(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_accuracy/validation", accuracy1, self.current_epoch)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


def main():
    #
    #print(len(LABEL_COLUMNS))
    N_EPOCHS = 10
    BATCH_SIZE = 8

    data_module = ToxicCommentDataModule(
        train_df,
        val_df,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT
    )

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5
    # warmup_steps, total_training_steps

    model = ToxicCommentTagger(
        n_classes=len(LABEL_COLUMNS),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    # from sklearn import metrics
    #
    # fpr = [0.        , 0.        , 0.        , 0.02857143, 0.02857143,
    #        0.11428571, 0.11428571, 0.2       , 0.4       , 1.        ]
    #
    # tpr = [0.        , 0.01265823, 0.67202532, 0.76202532, 0.91468354,
    #        0.97468354, 0.98734177, 0.98734177, 1.        , 1.        ]
    #
    # _, ax = plt.subplots()
    # ax.plot(fpr, tpr, label="ROC")
    # ax.plot([0.05, 0.95], [0.05, 0.95], transform=ax.transAxes, label="Random classifier", color="red")
    # ax.legend(loc=4)
    # ax.set_xlabel("False positive rate")
    # ax.set_ylabel("True positive rate")
    # ax.set_title("Example ROC curve")
    # plt.show()

    checkpoint_callback1 = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("../lightning_logs", name="toxic-comments7.8.1")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    trainer = pl.Trainer(
        logger=logger,
        # checkpoint_callback=checkpoint_callback1,
        callbacks=[early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=[],
        # progress_bar_refresh_rate=30,
        precision=16,
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
