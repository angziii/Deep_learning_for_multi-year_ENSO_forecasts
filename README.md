# Deep learning for multi-year ENSO forecasts

感谢***刘志强老师***的推荐与指导

我阅读的**第三篇**论文

由Google Gemini制作的**论文播客**下载：[AI vs. El Niño: Can Deep Learning Conquer the Climate's Toughest Puzzle?](https://github.com/angziii/Deep_learning_for_multi-year_ENSO_forecasts/blob/main/AI%20vs.%20El%20Nin%CC%83o%20%20Can%20Deep%20Learning%20Conquer%20the%20Climate's%20Toughest%20Puzzle.wav)

论文翻译：[ENSO-translation-cn](https://github.com/angziii/Deep_learning_for_multi-year_ENSO_forecasts/blob/main/ENSO-translation-cn.md)

[Read More](https://github.com/angziii/Deep_learning_for_multi-year_ENSO_forecasts/blob/main/present.md)

## 什么是ENSO？

El Niño-Southern Oscillation，埃尔尼诺-南方涛动。热带太平洋与大气之间的耦合系统，有三个状态，分别是El Niño和La Niña和它们的中间状态（中性）。热带太平洋处于 **La Niña->中性->El Niño->...** 的交替状态，每二至七年为一个周期。

上述两者是西班牙语中男孩和女孩的意思，El Niño源于圣诞节附近秘鲁海水变暖，指圣婴耶稣降临；La Niña为其反相现象，故称女孩。

El Niño 时，赤道东风减弱，暖海水回流南美沿岸，冷水上涌被抑制，导致海水变暖。La Niña 时，东风增强，暖水堆积西太平洋，东太平洋冷水上涌更强，导致海水异常变冷。

### 如何定量衡量ENSO？

用 **Nino3.4指数**，这是一块区域的 **Sea Surface Temperature(SST) Anomaly** 的平均值。

SST Anomaly = 当前值 - 历史平均值

这个区域位于热带太平洋，范围是 170∘−120∘W 经度范围和 5∘S−5∘N 纬度范围。（图源NASA）![image](https://github.com/user-attachments/assets/962d45b3-8754-42c4-9040-13c904b06b17)


### Nino3.4和ENSO的关系？

NOAA（美国国家海洋与大气管理局）标准：
	
 •	若 Nino3.4 区的 3 个月滑动平均 SST 异常值 ≥ +0.5°C，持续5个月以上，定义为 El Niño。

 •	若 ≤ -0.5°C，持续5个月以上，定义为 La Niña。

### 如何定义SST？

论文没有给出 SST 的精确测量深度，但通常是距离海面几米以内的温度。

### 只使用SST一个输入吗？

当然不是，SST是海洋与大气**直接**相互作用的体现，也是我们最直观感知和观测到的海洋状态。是ENSO事件的核心特征。

但是，海洋储存的总热量的变化，特别是在热带太平洋西部和中部，对于ENSO事件的**孕育、发展和衰减**至关重要。这就是海洋热容量（Heat Content, HC）,它代表了海洋一定深度范围内（在这篇论文中是上层300米）储存的总热量。

HC异常往往**领先于**东太平洋显著的SST异常，可以看作是海洋状态的一种“记忆”。同时使用两种输入，CNN模型能获得更全面的信息。

### 除了SST和HC，还有什么能作为输入吗？

不知道🤷‍♀️

## 这篇论文如何预测ENSO情况（和El Niño的类型）？

使用CNN模型。模型由input——conv1——MP1——conv2——MP2——conv3——FC layer——output，一共三个卷积层，两个最大池化层，一个全连接层组成。

我们输入过去三个月的地球海域的**SST Anomaly**和**HC**数据，输出是未来1到23个月的Nino3.4指数。

论文中也构造了另一个CNN模型，用于输出 EP、CP 和 Mix 型的概率，最终通过 softmax 归一化（我猜的）。

[***论文中用于ENSO预测的CNN网络详细讲解***](https://github.com/angziii/Deep_learning_for_multi-year_ENSO_forecasts/blob/main/CNN.md)

### 为什么用CNN？

标准的 CNN 通常包含卷积层、池化层，最后会接一个或多个全连接层，然后输出一个最终结果，比如一个数值（回归任务）或一组类别概率（分类任务）。

论文的主要预测目标是 Nino3.4 指数，这是**一个衡量 ENSO 强度的单一数值（回归任务）**。

另一个预测目标是 ENSO 类型（东太平洋型、中太平洋型、混合型），这是一个**分类任务**，最终需要输出属于各个类别的概率或最可能的类别。

### 为什么论文中的模型结构如此简单？

1. 它是一个**统计预测模型**，没有试图模拟复杂的物理过程；
2. 针对特定的任务设计，神经元和层数并不需要太多；
3. 简单的模型可解释性好（参见文中的**Heat Map**）。

### 训练中最大的困难是什么？

数据不足。可靠的全球海洋温度观测数据可用的历史时期相对较短，从1871年才开始。

这意味着对于每个特定的月份，可用于训练的样本数量**少于150个**。

### 如何解决数据不足的问题？

使用**transfer learning**生成数据初步训练 + 稀缺真实数据微调

首先，使用**CMIP5**的多个气候模型的大量历史**模拟数据**对CNN模型进行**初步训练**。CMIP5模型在一定程度上能逼真地模拟ENSO动态。

然后，使用1871年至1973年的再分析数据（结合了观测和模型的）对预训练的模型进行**fine tuning**。

### 训练中采取了什么特殊的策略？

训练数据和验证数据之间留出十年间隔（1974-1983年）的策略。

这个策略的主要目的是为了确保模型在验证期内的预测能力是真实的，而不是受到训练期内“海洋记忆”的虚假影响。

Q：“海洋记忆“如何影响模型？

### 损失函数如何构造的？

预测 Nino3.4 指数（回归任务）使用的是均方误差 (MSE) 作为损失函数。

预测厄尔尼奥类型（分类任务）很可能使用的是交叉熵损失，尽管论文没有明确命名，但其输出形式和任务性质符合使用交叉熵的场景。

### 学习率如何确定的？

论文中使用的学习率是 0.005，并且在训练过程中保持不变。可能是因为试出来效果最好。

### 如何衡量模型预测的效果？

用Correlation Skill。

Q：均方差不用吗？

### 如何构造Heat Map增强模型可解释性，并且标出重要贡献区域？

热力图旨在量化输入中各空间位置对预测结果的贡献。由于传统CNN在全连接层会丢失空间信息，热力图分析通过关注**最后一个卷积层的特征图与输出之间的连接权重**来恢复这种空间关联，从而揭示每个位置对最终预测的影响。

在标准的CNN前向传播中，从最后一个卷积层到全连接层时，会对卷积层特征图的空间维度（x和y）进行求和。但构建热力图时，论文明确说明避免了这种水平维度的求和。

Q：具体怎么计算还没看明白

## 取得了什么成果？

**显著提高了ENSO强度（Nino3.4指数）的长期预测能力**： 模型能够对ENSO进行长达一年半（18个月）的准确预测，其相关技巧在提前期超过6个月后系统性地优于所有最先进的动力预测系统。

**克服了传统模型在预测El Niño类型方面的弱点**： 模型在预测El Niño事件是东太平洋型、中太平洋型还是混合型方面取得了显著高于随机预测的命中率，而测试的动力预测模型在这方面表现不佳。

**更好地预测了海表温度的详细纬向分布**： 论文指出，CNN模型在这方面也优于动力预测模型。

### 除了上面这些呢？

This indicates that the CNN can be a powerful tool to **reveal complex ENSO mechanisms**. However, future studies are warranted to explore the physical mechanisms of the statistical relationship revealed by the CNN model with the limited sample size.

通过对模型的解释，我们能发现之前从未发现的影响 ENSO 的因素。

Q：这方面的物理机制研究开展了吗？
