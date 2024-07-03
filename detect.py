import numpy as np
import torch
from train import MyModel

if __name__ == '__main__':
    data_file = "dataset_0803.csv"
    # 导入数据集，前10列表示特征，最后一列表示分类
    xy = np.loadtxt(data_file, delimiter=',', dtype=np.float32)
    # x
    x_data = torch.Tensor(xy[:, 2:-1])
    # 实例化模型
    loaded_model = MyModel()
    # 加载模型参数
    loaded_model.load_state_dict(torch.load('./best_model.pth'))
    # 确保在评估模式下运行模型，这会关闭dropout等训练特定的层
    loaded_model.eval()
    # 假设我们有一些新的数据x_new
    x_new = torch.Tensor(x_data)  # new_data需要是一个np.array或者其他可转为torch.Tensor的数据类型
    # 使用加载的模型进行预测
    y_new_pred = loaded_model(x_new)
    # 将预测结果从Tensor转换为numpy数组
    y_new_pred = y_new_pred.detach().numpy()
    # 定义阈值
    threshold = 0.5
    # 将连续的预测值映射到二进制（0,1）值
    y_new_pred_binary = (y_new_pred > threshold).astype(int)
    print(y_new_pred_binary)
    # 将预测结果写入CSV文件
    np.savetxt("predictions.csv", y_new_pred_binary, delimiter=",")

