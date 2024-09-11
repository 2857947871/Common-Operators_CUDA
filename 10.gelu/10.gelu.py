import numpy as np
import matplotlib.pyplot as plt

# GELU 激活函数的实现
def gelu(x):
    alpha = 0.7978845608028654
    beta = 0.044714998453855515
    return 0.5 * x * (1 + np.tanh(alpha * x + beta * x**3))

# 创建输入数据
x = np.linspace(-3, 3, 1000)
y = gelu(x)

# 绘制函数图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='GELU Activation Function', color='b')
plt.title('GELU Activation Function')
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.grid(True)
plt.legend()
plt.show()