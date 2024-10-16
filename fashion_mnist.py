import tensorflow as tf
import numpy as np
import pandas as pd

# 加载 Fashion MNIST 数据集
train_df = pd.read_csv('data/fashion-mnist_train.csv')
test_df = pd.read_csv('data/fashion-mnist_test.csv')

# 提取数据和标签
x_train = train_df.iloc[:, 1:].values  # 图像像素数据
y_train = train_df.iloc[:, 0].values   # 标签数据

x_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# 数据归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 将数据 reshape 为 28x28 的图像
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# 将标签转换为二进制分类问题
binary_class_train_indices = np.where((y_train == 0) | (y_train == 1))
binary_class_test_indices = np.where((y_test == 0) | (y_test == 1))

x_train_binary = x_train[binary_class_train_indices]
y_train_binary = y_train[binary_class_train_indices]
x_test_binary = x_test[binary_class_test_indices]
y_test_binary = y_test[binary_class_test_indices]

y_train_binary = np.where(y_train_binary == 0, 0, 1)
y_test_binary = np.where(y_test_binary == 0, 0, 1)

# 构建多层神经网络
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型，使用 Adam 优化器
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 使用 mini-batch 训练模型
# 通过指定 batch_size 参数来实现 mini-batch 训练
model.fit(x_train_binary, y_train_binary, epochs=10, batch_size=32, validation_data=(x_test_binary, y_test_binary))

# 评估模型
test_loss, test_acc = model.evaluate(x_test_binary, y_test_binary)
print(f"测试准确率：{test_acc}")
