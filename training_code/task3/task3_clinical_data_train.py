import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import random
from util import *

set_seed(42)
save_dir = './models/'
csv_path = 'save_csv'
if not os.path.exists(csv_path):
    os.mkdir(csv_path)

df = pd.read_csv("/mnt/SSD3/xulin/phd_research/chimera/code/train_test_split/seed3/task3/train_dataset_new.csv")
df.replace({-1: np.nan, "-1": np.nan}, inplace=True)

X = df.drop(columns=["patient_id", "progression", "Time_to_prog_or_FUend", 'label'])
# y_train = df["label"].astype(int)
train_id = df['patient_id'].tolist()
event = df['progression'].tolist()
converted_event = [1 if x == 0 else 0 for x in event]
survival_time = df['Time_to_prog_or_FUend'].tolist()
label = df['label'].tolist()

# test csv
df_test = pd.read_csv("/mnt/SSD3/xulin/phd_research/chimera/code/train_test_split/seed3/task3/test_dataset.csv")
df_test.replace({-1: np.nan, "-1": np.nan}, inplace=True)

test_feat = df_test.drop(columns=["patient_id", "progression", "Time_to_prog_or_FUend", 'label'])
# y_test = df_test["BRS_binary"].astype(int)
test_id = df_test['patient_id'].tolist()
event_test = df_test['progression'].tolist()
converted_event_test = [1 if x == 0 else 0 for x in event_test]
survival_time_test = df_test['Time_to_prog_or_FUend'].tolist()
label_test = df_test['label'].tolist()

numerical_features = ["age", "no_instillations"]
categorical_features = X.select_dtypes(include="object").columns.tolist()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    # ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(test_feat)

joblib.dump(preprocessor, "clinical_preprocessor.joblib")

class Main_Dataset(Dataset):
    def __init__(self, X, train_id, train_converted_event, train_survival_time, train_label):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.file_list = train_id
        self.survival_time = train_survival_time
        self.censor_status = train_converted_event
        self.label = train_label
        # self.target_sequence_num = target_sequence_num

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, ind):
        id_name = self.file_list[ind]
        survival_time = self.survival_time[ind]
        censor_status = self.censor_status[ind]
        label = self.label[ind]

        info = torch.LongTensor([ind]).squeeze()
        event_time = torch.FloatTensor([survival_time]).squeeze()
        c = torch.FloatTensor([censor_status]).squeeze()
        label = torch.LongTensor([label]).squeeze()

        return self.X[ind], label, event_time, c, info

train_dataset = Main_Dataset(X_train, train_id, converted_event, survival_time, label)
test_dataset = Main_Dataset(X_test, test_id, converted_event_test, survival_time_test, label_test)
print('the number of training data: ', len(train_dataset))
print('the number of test data: ', len(test_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 4),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        hazards = torch.sigmoid(x)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S

input_dim = X_train.shape[1]
model = MLPBinaryClassifier(input_dim)
model = model.to(device="cuda")
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = NLLSurvLoss(alpha=0)
epoch_lst, train_loss_lst, train_cindex_lst, test_cindex_lst = [], [], [], []

num_epochs = 90

best_cindex = 0.0
# best_model_path = "best_model.pt"

for epoch in range(num_epochs):
    epoch_lst.append(epoch)
    model.train()
    step = 0
    total_loss = 0
    for i, data in enumerate(train_loader):
        ret_feat, label, event_time, c, info = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[
            4].cuda()
        batch_size = ret_feat.size(0)

        optimizer.zero_grad()

        hazards, S = model(ret_feat)
        # print('pred risk size: ', pred_risk.size())

        _, loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        risk = -torch.sum(S, dim=1)  # .detach().cpu().numpy()

        if step % 3 == 0:
            print(
                'epoch:{} - step:{} - train survival loss: {:.3f}'.format(
                    epoch,
                    step,
                    loss_micro.item()))

        loss_micro.backward()
        total_loss += loss_micro.item()
        optimizer.step()

        step += 1
        # progress_bar(i, len(train_loader), 'train')
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    train_loss_lst.append(avg_loss)

    # net_state_dict = model.module.state_dict()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save({
        'epoch': epoch,
        'train_survival_loss': loss_micro},
        # 'net_state_dict': net_state_dict
        os.path.join(save_dir, '%03d.ckpt' % epoch))
    torch.save(model.state_dict(), os.path.join(save_dir, '%03d.pt' % epoch))
    # train_loss_lst.append("%.3f" % loss_micro.data.item())

    train_cindex, _ = evaluate(model, train_loader)
    print('training c-index: %.3f' % train_cindex)
    # print(f"Test ACC: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
    train_cindex_lst.append(train_cindex)

    test_cindex, preds_list = evaluate(model, test_loader)
    print('test c-index: %.3f' % test_cindex)
    # print(f"Test ACC: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
    test_cindex_lst.append(test_cindex)

    df_test_results = df_test.copy()
    df_test_results["pred_risk"] = preds_list

    df_test_results.to_csv(os.path.join(csv_path, '%03d_infer_per_patient.csv' % epoch), index=False)


result = pd.DataFrame()
result['epoch'] = epoch_lst
result['train_loss'] = train_loss_lst
result['train_cindex'] = train_cindex_lst
result['test_cindex'] = test_cindex_lst

result = result.round({
    'train_loss': 3,
    'train_cindex': 3,
    'test_cindex': 3
})

result.to_csv(os.path.join(csv_path, 'clinical_data_model_result.csv'), index=None)
print('finishing training')

epochs = list(range(1, len(train_loss_lst) + 1))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_lst, label="Train Loss", color="tab:red", marker="o", markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_cindex_lst, label="Train C-index", color="tab:blue", marker="o", markersize=4)
plt.plot(epochs, test_cindex_lst, label="Test C-index", color="tab:green", marker="s", markersize=4)
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("C-index over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()