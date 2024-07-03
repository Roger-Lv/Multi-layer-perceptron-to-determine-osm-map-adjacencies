import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss, Linear, ReLU, Sigmoid, BatchNorm1d
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# 建立新模型，现在有3个线性层，并使用ReLU激活函数
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = Linear(20, 24, bias=True)
        self.linear2 = Linear(24, 40, bias=True)
        self.linear3 = Linear(40, 60, bias=True)
        self.linear4 = Linear(60, 40, bias=True)
        self.linear5 = Linear(40, 20, bias=True)
        self.linear6 = Linear(20, 5, bias=True)
        self.linear7 = Linear(5, 1, bias=True)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = torch.relu(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        return x



def calculate_accuracy(y_pred, y_target):
    """计算准确率"""
    predicted = (y_pred > 0.5).float()
    accuracy = (predicted == y_target).sum().item() / len(y_target)
    return accuracy


if __name__ == "__main__":
    # 导入数据集，前10列表示特征，最后一列表示分类
    xy = np.loadtxt("dataset_0803.csv", delimiter=',', dtype=np.float32)
    length = 10000
    epochs = 2000
    # 取第1000行及之后的所有数据作为训练集
    x_data = torch.Tensor(xy[length:, 2:-1])
    y_data = torch.Tensor(xy[length:, [-1]])

    # 创建数据加载器
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    # 类实例化
    my_model = MyModel()

    # 二分类问题，采用BCELoss
    loss_cal = BCELoss()

    # 使用Adam优化器，lr设置为0.02
    optimizer = Adam(my_model.parameters(), lr=0.01)

    # 定义存储损失函数值的列表
    epoch_list = []
    loss_list = []

    # 450轮训练
    best_accuracy = 0.0
    best_loss = 10000
    count_loss = 0

    # 在主循环开始前，初始化一个字典来保存每个epoch的评估指标：
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for epoch in range(epochs):
        print(f"\nEpoch {epoch} starts!")
        losses = []
        accuracies = []

        for batch_x, batch_y in dataloader:  # 在每个批次上进行训练
            y_pred = my_model(batch_x)
            loss = loss_cal(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算并记录损失和准确率
            losses.append(loss.item())
            accuracies.append(calculate_accuracy(y_pred.data, batch_y))

        # 每个Epoch结束后，计算平均损失和准确率
        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")

        # 在损失列表中记录每个Epoch的平均损失
        loss_list.append(avg_loss)
        epoch_list.append(epoch)

        # 如果此epoch的平均精度超过之前的最佳精度，则保存模型
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            torch.save(my_model.state_dict(), 'best_model.pth')
        # 计算整个训练集的预测值
        y_pred_full = my_model(x_data).detach()
        predicted_full = (y_pred_full > 0.5).float().numpy()
        # 计算各种评价指标并保存到字典中
        metrics["accuracy"].append(accuracy_score(y_data.numpy(), predicted_full))
        metrics["precision"].append(precision_score(y_data.numpy(), predicted_full))
        metrics["recall"].append(recall_score(y_data.numpy(), predicted_full))
        metrics["f1"].append(f1_score(y_data.numpy(), predicted_full))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(my_model.state_dict(), 'best_model.pth')
            count_loss = 0
        else:
            count_loss += 1
        if count_loss > 600:
            break

    # 显示损失函数随训练轮数的变化
    plt.figure()
    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 在所有epoch结束后，找到准确率最高的epoch
    best_epoch = np.argmax(metrics["accuracy"])

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Training Precision: {metrics['precision'][best_epoch]:.2f}")
    print(f"Best Training Recall: {metrics['recall'][best_epoch]:.2f}")
    print(f"Best Training F1 Score: {metrics['f1'][best_epoch]:.2f}")
    print(f"Best Training Accuracy: {metrics['accuracy'][best_epoch] * 100:.2f}%")

    # 输出训练集上的准确率
    y_pred_train = my_model(x_data)
    predicted_train = (y_pred_train.data > 0.5).float()
    accuracy_train = (predicted_train == y_data).sum().item() / len(y_data)
    print('Accuracy on Training Set: {:.2f}%'.format(accuracy_train * 100))
    # 计算训练集上的预测值
    predicted_train = (y_pred_train.data > 0.5).float().numpy()

    # 计算各种评价指标
    precision_train = precision_score(y_data.numpy(), predicted_train)
    recall_train = recall_score(y_data.numpy(), predicted_train)
    f1_train = f1_score(y_data.numpy(), predicted_train)
    accuracy_train = accuracy_score(y_data.numpy(), predicted_train)

    print(f"Training Precision: {precision_train:.2f}")
    print(f"Training Recall: {recall_train:.2f}")
    print(f"Training F1 Score: {f1_train:.2f}")
    print(f'Training Accuracy: {accuracy_train * 100:.2f}%')

    # 取前五行的所有数据作为测试集
    x_test = torch.Tensor(xy[:length, 2:-1])
    y_test = torch.Tensor(xy[:length, [-1]])

    y_test_pred = my_model(x_test)

    # 计算测试集上的预测值
    predicted_test = (y_test_pred.data > 0.5).float().numpy()

    # 计算各种评价指标
    precision_test = precision_score(y_test.numpy(), predicted_test)
    recall_test = recall_score(y_test.numpy(), predicted_test)
    f1_test = f1_score(y_test.numpy(), predicted_test)
    accuracy_test = accuracy_score(y_test.numpy(), predicted_test)

    print(f"\nTesting Precision: {precision_test:.2f}")
    print(f"Testing Recall: {recall_test:.2f}")
    print(f"Testing F1 Score: {f1_test:.2f}")
    print(f'Testing Accuracy: {accuracy_test * 100:.2f}%')

    # 实例化一个新的模型对象
    best_model = MyModel()
    # 加载准确率最高的模型参数
    best_model.load_state_dict(torch.load('best_model.pth'))

    # 用最好的模型预测测试集
    y_test_pred_best = best_model(x_test)

    # 计算测试集上的预测值
    predicted_test_best = (y_test_pred_best.data > 0.5).float().numpy()

    # 计算各种评价指标
    precision_test_best = precision_score(y_test.numpy(), predicted_test_best)
    recall_test_best = recall_score(y_test.numpy(), predicted_test_best)
    f1_test_best = f1_score(y_test.numpy(), predicted_test_best)
    accuracy_test_best = accuracy_score(y_test.numpy(), predicted_test_best)

    print(f"\nTesting Precision with Best Model: {precision_test_best:.2f}")
    print(f"Testing Recall with Best Model: {recall_test_best:.2f}")
    print(f"Testing F1 Score with Best Model: {f1_test_best:.2f}")
    print(f'Testing Accuracy with Best Model: {accuracy_test_best * 100:.2f}%')
