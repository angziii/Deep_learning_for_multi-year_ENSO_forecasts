**深度学习用于多年ENSO预测**

Yoo-Geun Ham\*, Jeong-Hwan Kim¹ & Jing-Jia Luo²³

\*通讯作者：ygham@jnu.ac.kr

**摘要**

厄尔尼奥-南方涛动（ENSO）的变化与广泛的区域气候极端事件和生态系统影响相关¹。因此，可靠的、长提前期的预测对于管理政策响应将非常有价值。但尽管经过数十年的努力，预测提前期超过一年的ENSO事件仍然存在问题²。在这里，我们展示了一个采用深度学习方法的统计预测模型，能够对ENSO进行长达一年半的准确预测。为了规避观测数据量有限的问题，我们使用迁移学习，首先在历史模拟数据上训练一个卷积神经网络（CNN），然后在使用1871年至1973年的再分析数据进行训练。在1984年至2017年的验证期内，CNN模型的Nino3.4指数的全季节相关技巧远高于当前最先进的动力预测系统。CNN模型在预测海表温度的详细纬向分布方面也表现更好，克服了动力预测模型的弱点。热力图分析表明，CNN模型使用物理上合理的前兆信号预测ENSO事件。因此，CNN模型是ENSO事件预测及其相关复杂机制分析的强大工具。

预测大规模气候变化的能力，及其对全球社会和环境系统的影响，高度依赖于ENSO预测的质量。尽管使用海气耦合模型的ENSO预测⁵'⁶通常优于当前的统计模型¹，但最先进的动力预测系统仍然无法对提前期超过一年的ENSO进行准确预测。因此，多年ENSO事件的预测仍然是一个重大挑战¹。

然而，ENSO中存在一个振荡元素，与缓慢变化的海洋变化及其与大气的耦合有关，这表明多年预测是可能的。有趣的是，几次拉尼娜事件期间赤道太平洋的异常持续了几年⁸。高频赤道风的可预测性较低，但与底层海表温度耦合的缓慢变化的赤道风分量¹⁰在一定程度上是可预测的。赤道太平洋以外的SST异常可能导致ENSO事件，其时间滞后超过一年¹¹'¹²。这些研究表明，ENSO预测仍有改进空间，尽管现有方法可能不适用。

随着大数据时代的到来，深度学习通过发现大型数据集中复杂的结构¹³对许多领域产生了巨大影响。特别是，CNN在处理具有空间结构的多维数组数据（例如，用于图像中物体的识别¹⁴'¹⁵）方面取得了突出成果。因此，CNN适用于揭示三维预测因子场与被预测指数之间的联系。在这里，我们使用基于CNN的统计模型来预测ENSO指数。

我们的CNN模型使用 0∘−360∘E, 55∘S−60∘N 范围内连续三个月的SST和热含量（上层300米海洋温度的垂直平均）异常图作为预测因子，并使用Nino3.4指数（170∘−120∘W, 5∘S−5∘N 区域平均SST异常）作为被预测量，提前预测长达两年（图1，方法）。

将深度学习应用于气候预测的最大限制之一是观测期太短，无法实现适当的训练。全球海洋温度分布的观测数据可追溯到1871年¹⁶。这意味着，对于每个日历月，样本数量少于150个。为了大幅增加训练数据量，我们利用了参与耦合模式比较项目第五阶段（CMIP5）的气候模型的输出，其中ENSO在一定程度上得到了逼真的模拟¹⁷（扩展数据表1）。1871年至1973年的再分析数据也用于训练CNN模型¹⁶。验证预测技巧的时期是1984年至2017年（扩展数据表2）。我们在训练期最后一年与验证期最早一年之间留出十年的间隔，以消除训练期中海洋记忆对验证期ENSO可能产生的影响。

我们应用迁移学习技术¹⁸，利用CMIP5输出和再分析数据对CNN进行最优训练。该技术利用从具有大量样本的类似任务中获得的知识来执行目标任务。在本研究中，CNN模型首先使用CMIP5输出进行训练，然后使用训练好的权重作为初始权重来构建最终的CNN模型，并使用再分析数据进行微调。CNN中反映CMIP5样本系统误差的部分在第二次使用再分析数据训练后得到修正。

图2a显示了1984年至2017年三年移动平均Nino3.4指数的全季节相关技巧。CNN模型中Nino3.4指数的预测技巧在提前期超过六个月时系统性地优于所有最先进的动力预测系统。CNN模型是前六个月预测提前期内预测技巧最好的两个模型之一。CNN模型中Nino3.4指数的全季节相关技巧在提前期长达17个月时高于0.5，而在领先的动力预测系统SINTEX-F³中，17个月提前期的相关技巧为0.37。我们得出结论，CNN模型能够对ENSO事件进行长达一年半的准确预测：这是任何最先进预测系统都无法实现的结果。除了深度学习算法相对于以前的统计方法的优越性（扩展数据图1和扩展数据表3）之外，CNN模型预测技巧的提高归因于大量的CMIP5样本以及迁移学习技术的成功应用¹⁸（扩展数据图2）。

CNN模型的技巧受训练数据集变化的影响不大（补充图1）。CNN模型甚至在预测一些能够捕捉逼真ENSO动态的CMIP5模型中的模拟ENSO指数方面取得了成功（提前期长达1.5年时相关技巧超过0.5）（补充图2-4）。改变训练集和验证集产生的技巧不确定性很小，表明CNN可以提供准确的实时预测。

与SINTEX-F相比（图2b，c），CNN模型在几乎所有目标季节都显示出更高的Nino3.4指数相关技巧。对于北半球春末到秋季之间的季节预测，相关技巧的提高尤为显著。例如，SINTEX-F对五月-六月-七月（MJJ）季节的预测，相关技巧仅在提前期长达四个月时超过0.5，而CNN则可达11个月。这缩小了CNN在不同目标季节之间的预测技巧差距，我们得出结论，CNN模型受春季可预测性障碍¹⁹的影响较小。

18个月提前期预测的十二月-一月-二月（DJF）季节的Nino3.4指数表明，CNN模型正确预测了ENSO的振幅（图3a）。为了理解CNN模型如何能够成功预测如此长提前期的Nino3.4指数，我们对1997/98年厄尔尼奥事件的18个月提前期预测生成了热力图²⁰（图3b）。热力图量化了输入数据中每个网格点对被预测量的贡献；热力图中的正值（负值）表示某些区域的预测因子对预测正（负）Nino3.4有贡献（方法）。1996年MJJ季节的预测因子（异常）对1997/98年DJF季节的热力图表明，热带西太平洋²¹、印度洋¹¹和副热带大西洋¹²（图3b中的红色阴影区域）的异常是成功预测1997/98年厄尔尼奥的主要贡献者。

热力图突出显示的海洋信号可以基于物理联系引发1997/98年的厄尔尼奥。南热带西太平洋的正热含量异常表示连续厄尔尼奥发展所需的补充热量²¹。西南印度洋的负SST异常导致1996年北半球秋季出现负印度洋偶极子（IOD）¹¹（扩展数据图3a和4b）。负IOD导致随后的季节整个印度洋出现负SST异常（扩展数据图3b和4d），这触发了赤道西太平洋的西风，从而在一年后引发厄尔尼奥事件²²。此外，1996年MJJ季节北副热带大西洋的负SST异常通过激发中纬度太平洋变化¹²（扩展数据图3c，d和4a）对1997/98年厄尔尼奥事件有贡献。

除了ENSO振幅之外，厄尔尼奥事件的全球影响因厄尔尼奥SST异常的详细纬向分布而差异很大：中太平洋型（CP-type）和东太平洋型（EP-type）厄尔尼奥²³。因此，基于SST异常纬向位置成功预测厄尔尼奥类型对于提高全球气候预测质量至关重要。为此，我们建立了另一个CNN模型来预测厄尔尼奥类型。在该模型中，被预测量对应于三种厄尔尼奥类别的发生百分比²⁴：CP型、EP型和两者的混合型。三种类型中概率最高的类别被视为最终预测。我们注意到，再分析数据没有用于训练预测厄尔尼奥类型的CNN模型，因为已知再分析训练期内的厄尔尼奥事件属于单一类型²³。因此，我们仅使用CMIP5模型的输出训练CNN模型来预测厄尔尼奥类型，并且不应用迁移学习技术。

进行了一系列回溯预测实验，以提前12个月预测厄尔尼奥事件的类型，CNN模型的命中率在验证期（1984-2017年）为66.7%（图4a和扩展数据表4）。随机预测在95%置信区间内的命中率在12.5%到62.5%之间，因此CNN的66.7%命中率显著优于随机预测，P值为0.016。相比之下，没有一个动力预测模型显示出比随机预测在统计学上显著更好的预测技巧，这意味着CNN模型克服了最先进预测模型的长期弱点²⁵。这表明基于深度学习的模型可以高精度地预测厄尔尼奥事件的空间复杂性¹⁶'²⁶。

除了预测ENSO强度和类型外，CNN模型还允许我们识别哪些SST信号导致EP型或CP型厄尔尼奥事件。为此，我们计算了五个海洋盆地（扩展数据图5）的区域平均热力图值。然后，我们为每个海洋盆地选择热力图值最大的两个案例，这可以被认为是EP型或CP型厄尔尼奥事件发展最有利的模式。

尽管一些模式是从混合型厄尔尼奥发生的年份中选取的，但现有文献和额外分析表明，选定的模式可能导致EP型或CP型厄尔尼奥。对于EP型厄尔尼奥，目标季节前一年选定的热含量异常可以引发厄尔尼奥成熟期前一个季节的正IOD²⁷（图4e）。对于CP型厄尔尼奥，显示了北热带大西洋的SST降温（图4f，g）。我们的结果与之前的研究¹¹'²⁸一致。南太平洋和印度洋的CP型厄尔尼奥前兆此前未见报道，额外分析表明，识别出的前兆可能导致CP型厄尔尼奥事件（扩展数据图6）。这表明CNN可以成为揭示复杂ENSO机制的强大工具。然而，需要未来的研究来探索CNN模型在有限样本量下揭示的统计关系的物理机制。

CNN优于以前模型的原因在于通过卷积过程成功提取了输入变量中的特征。CNN识别可以用于编码各种不同形状的基本形状，从而表现出对平移和形变的局部不变性²⁹。因此，在CNN内部，即使详细的空间分布与典型前兆信号发生了偏移或形变，前兆信号也能正确地影响被预测量。此外，卷积过程使得CNN模型可以使用相对较少的气候样本进行适当训练。

深度学习在预测许多地球系统分量方面取得了进展，但迄今为止其在气候预测中的应用仍然罕见³⁰。本文报道的通过迁移学习和热力图分析成功应用深度学习预测和理解气候现象，可以促进工程学和地球科学之间的跨学科研究。

**在线内容**

任何方法、补充参考文献、Nature Research报告摘要、源数据、扩展数据、补充信息、致谢、同行评审信息；作者贡献和竞争利益的详细信息；以及数据和代码可用性声明可在[https://doi.org/10.1038/s41586-019-1559-7获取](https://doi.org/10.1038/s41586-019-1559-7获取)。

收稿日期：2018年11月20日；接受日期：2019年7月10日；  
在线发表日期：2019年9月18日。  
**参考文献**

1. McPhaden, M. J., Zebiak, S. E. & Glantz, M. H. ENSO as an integrating concept in Earth science. Science 314, 1740-1745 (2006).  
2. Barnston, A. G., Tippett, M. K, L'Heureux, M. L, LI, S. & DeWitt, D. G. Skill of real-time seasonal ENSO model predictions during 2002-11: is our capability increasing? Bull. Am. Meteorol. Soc. 93, 631-651 (2012).  
3. Taylor, K. E., Stouffer, R. J. & Meehl, G. A. An overview of CMIP5 and the experiment design. Bull. Am. Meteorol. Soc. 93, 485-498 (2012).  
4. Cane, M. A., Zebiak, S. E. & Dolan, S. C. Experimental forecasts of El Niño. Nature 321, 827-832 (1986).  
5. Luo, J.-J., Masson, S., Behera, S. K. & Yamagata, T. Extended ENSO predictions using a fully coupled ocean-atmosphere model. J. Clim. 21, 84-93 (2008).  
6. Tang, Y. et al. Progress in ENSO prediction and predictability study. Nati Sci. Rev. 5, 826-839 (2018).  
7. Chen, D., Cane, M. A., Kaplan, A., Zebiak, S. E & Huang, D. Predictability of El Niño over the past 148 years. Nature 428, 733-736 (2004).  
8. Gao, C. & Zhang, R. H. The roles of atmospheric wind and entrained water temperature (Te) in the second-year cooling of the 2010-12 La Niña event. Clim. Dyn. 48, 597-617 (2017).  
9. Hu, S. & Fedorov, A. V. Exceptionally strong easterly wind burst stalling El Niño of 2014\. Proc. Natl Acad. Sci. USA 113, 2005-2010 (2016).  
10. Gebbie, G. & Tziperman, E. Predictability of SST-modulated westerly wind bursts. J. Clim. 22, 3894-3909 (2009).  
11. Izumo, T. et al. T. Influence of the state of the Indian Ocean Dipole on the following year's El Niño. Nat. Geosci. 3, 168-172 (2010).  
12. Park, J. H., Kug, J. S., Li, T. & Behera, S. K. Predicting El Niño beyond 1-year lead: effect of the Western Hemisphere warm pool. Sci. Rep. 8, 14957 (2018).  
13. LeCun, Y., Bengio, Y., & Hinton, G. Deep learning. Nature 521, 436-444 (2015).  
14. Krizhevsky, A., Sutskever, I. & Hinton, G. E. Imagenet classification with deep convolutional neural networks. Adv. Neural Inf. Process Syst. 25, 1097-1105 (2012).  
15. Oquab, M., Bottou, L., Laptev, I. & Sivic, J. Learning and transferring mid-level image representations using convolutional neural networks. In Proc. IEEE Conf. on Computer Vision And Pattern Recognition 1717-1724 (IEEE, 2014).  
16. Giese, B. S. & Ray, S. El Niño variability in simple ocean data assimilation (SODA), 1871-2008. J. Geophys. Res. Oceans 116, https://doi. org/10.1029/2010JC006695 (2011).  
17. Bellenger, H., Guilyardi, E., Leloup, J., Lengaigne, M. & Vialard, J. ENSO representation in climate models: from CMIP3 to CMIP5. Clim. Dyn. 42, 1999-2018 (2014).  
18. Yosinski, J., Clune, J., Bengio, Y. & Lipson, H. How transferable are features in deep neural networks? Adv. Neural Inf. Process. Syst. 27, 3320-3328 (2014).  
19. Webster, P. J. & Yang, S. Monsoon and ENSO: selectively interactive systems. Q. J. R. Meteorol. Soc. 118, 877-926 (1992).  
20. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A. & Torralba, A. Learning deep features for discriminative localization. In Proc. IEEE Conference on Computer Vision and Pattern Recognition 2921-2929 (IEEE, 2016).  
21. Anderson, B. T. On the joint role of subtropical atmospheric variability and equatorial subsurface heat content anomalies in initiating the onset of ENSO events. J. Clim. 20, 1593-1599 (2007).  
22. Kug, J. S. & Kang, I. S. Interactive feedback between ENSO and the Indian Ocean. J. Clim. 19, 1784-1801 (2006).  
23. Yeh, S. W. et al. El Niño in a changing climate. Nature 461, 511 (2009).  
24. Zhang, Z., Ren, B. & Zheng, J. A unified complex index to characterize two types of ENSO simultaneously. Sci. Rep. 9, 8373 (2019).  
25. Pillai, P. A. et al. How distinct are the two flavors of El Niño in retrospective forecasts of Climate Forecast System version 2 (CFSv2)? Clim. Dyn. 48, 3829-3854 (2017).  
26. Johnson, N. C. How many ENSO flavors can we distinguish? J. Clim. 26, 4816-4827 (2013).  
27. Webster, P. J., Moore, A. M., Loschnigg, J. P. & Leben, R. R. Coupled ocean-atmosphere dynamics in the Indian Ocean during 1997-98. Nature 401, 356-360 (1999).  
28. Ham, Y. G., Kug, J. S. & Park, J. Y. Two distinct roles of Atlantic SSTs in ENSO variability: north tropical Atlantic SST and Atlantic Niño. Geophys. Res. Lett. 40, 4012-4017 (2013).  
29. Zeiler, M. D. & Fergus, R. Visualizing and understanding convolutional networks. In Eur. Conf. On Computer Vision 818-833 (Springer, 2014\)  
30. Reichstein, M. et al. Deep learning and process understanding for data-driven Earth system science. Nature 566, 195-204 (2019).  
31. Hunter, J. D. Matplotlib: a 2D graphics environment. Comput. Sci. Eng. 9, 90-95 (2007).

**出版商说明** Springer Nature对已发表地图和机构附属关系的管辖权主张保持中立。

© 作者，在Springer Nature Limited的独家许可下，2019年

**方法**

**应用于ENSO预测的CNN模型架构。** 用于ENSO预测的基于CNN³²的统计模型包含三个卷积层和两个最大池化层。最大池化过程从每个2×2网格中提取最大值。第三个卷积层连接到全连接层中的神经元，全连接层连接到最终输出。输出的维度为一，并且针对每个预测提前期和目标季节分别构建CNN模型。卷积核和全连接层中神经元的总数分别为30或50。因此，CNN模型有四种组合（C30H30、C30H50、C50H30和C50H50，其中C和H后面的数字分别表示卷积核和全连接层中神经元的数量）。C30H30的总参数数量为117,511，C30H50为182,351，C50H30为211,811，C50H50为319,851。对不同数量卷积核和神经元的四个CNN模型的预测Nino3.4指数进行平均，以获得最终预测结果。这种平均通过抵消单个CNN模型中的预测误差，导致预测技巧略有系统性提高。每个epoch的mini batch大小设置为400，使用CMIP5输出进行第一次训练的epoch数量为700。epoch数量从600到1000的变化对ENSO预测技巧没有影响。对于使用再分析数据进行的第二次训练，epoch数量设置为20。学习率固定为0.005，并且在整个迭代过程中没有改变：未使用学习率调度。

**卷积过程。** CNN的卷积过程涉及从全局地图中提取局部特征，以及计算卷积核中的值与输入层中的值之间的点积。卷积过程的输出然后转化为特征图。卷积核的值通过迭代自动确定，以最小化成本函数，成本函数定义为预测分布和真实分布之间的均方差。

第i个卷积层中第j个特征图在网格点(x, y)的值（表示为 \(v_{i,j}^{x,y}\)）使用以下公式计算：

\[
v_{i,j}^{x,y}= \tanh\!\left(
    \sum_{m=1}^{M_{i-1}}
    \sum_{p=1}^{P_i}
    \sum_{q=1}^{Q_i}
    w_{i,j,m}^{p,q}\;
    \nu_{i-1,m}\!\left(x+p-\frac{P_i}{2},\,y+q-\frac{Q_i}{2}\right)
    + b_{i,j}
\right)
\]

其中Pi和Qi分别表示第i个卷积层的卷积核的纬向和经向维度。使用双曲正切函数(tanh)作为激活函数。第一次卷积过程（即P1=8; Q1=4)的卷积核维度设置为8×4，第二次和第三次卷积过程设置为4×2。Mi-1表示第(i-1)个层中特征图的数量。另一方面，wi,j,m^{p,il}表示卷积核中网格点(p, q)处的权重；这用于将第(i-1)个层中的第m个特征图连接到第i个卷积层中的第j个特征图。此外，y(x+p-Ps/2y+q-Qs/2)表示第(i−1)个卷积层中第(−1,m)个特征图在网格点(x+p-Pi/2,y+q-Qi/2)的值，而bi,i表示第i个卷积层中第j个特征图的偏置。为了确保第i个层的水平维度与第(i-1)个层的水平维度对应，使用填充技术用零填充空白空间。

**热力图分析。** 为了说明热力图的计算，先给出输出变量：

\[
V = \sum_{n=1}^{N} \tanh\!\left(
        \sum_{m=1}^{M_L}\sum_{y=1}^{Y_L}\sum_{x=1}^{X_L}
        W_{F,m,n}^{x,y}\,v_{L,m}^{x,y} + b_{F,n}
    \right) W_{O,n} + b_{O}
\]

其中V表示输出神经元（即被预测量），而XL和YL表示第三个卷积层中特征图的维度（即XL=18; Y1=6)。N表示全连接层中神经元的数量，WF,m,n^{x,y}表示网格点(x, y)处的权重（用于将最后一个卷积层L中的第i个特征图连接到全连接层F中的第n个神经元），vL,m^{x,y}表示最后一个卷积层L中第m个特征图在网格点(x, y)的值，bF,n表示全连接层F中第sth个神经元的偏置；WO,n表示权重（用于将全连接层中的第i个神经元连接到输出层O），b0表示输出层O的偏置。通过这个过程，我们计算最后一个卷积层和权重之间乘积的总和；因此，所有空间信息都丢失了。

然而，热力图的计算考虑了每个网格点对输出神经元的贡献：这由用于连接最后一个卷积层到全连接层的权重（即WF,m,nx,y）表示。换句话说，热力图的计算避免了水平维度的求和。输出层神经元在网格点(x, y)处的热力图值（表示为hx,y）最终使用以下公式计算：

\[
h^{x,y}= \sum_{n=1}^{N} \tanh\!\Bigl(
    \bigl|\,\sum_{m=1}^{M_L} W_{E,m,n}^{x,y}\,v_{L,m}^{x,y}
          + X_LY_L\,b_{E,n}\bigr|
\Bigr) W_{O,n} + X_LY_L\,b_{O}
\]

**ENSO指数预测。** Nino3.4指数（170∘−120∘W, 5∘S−5∘N 区域平均SST异常）在本研究中用作被预测量。预测期的开始对应于最新可用观测数据的时间。例如，与使用前一年OND时期预测因子的CNN模型预测相比，从1月1日开始的动力预测系统的输出。提前期定义为最新可用观测数据与三个月预测目标期中间之间相隔的月数。所有预测提前期的目标期都包含在1984年1月至2017年12月之间。

使用时间异常相关系数 \(C\)（函数于提前期 \(l\)）评估 Nino3.4 指数预测技巧：

\[
C_l = \sum_{m=1}^{12}
      \frac{
          \displaystyle\sum_{y=s}^{e}
          \!\left(Y_{y,m}-\overline{Y}_m\right)
          \left(P_{y,m,l}-\overline{P}_{m,l}\right)
      }{
          \sqrt{\displaystyle\sum_{y=s}^{e}
                \!\left(Y_{y,m}-\overline{Y}_m\right)^2}\;
          \sqrt{\displaystyle\sum_{y=s}^{e}
                \!\left(P_{y,m,l}-\overline{P}_{m,l}\right)^2}
      }
\]

这里，Y和P分别表示观测值和预测值。Yn和Pim,l分别表示相对于日历月m（从1到12）和预测提前期l的时间气候平均值。标签y表示预测目标年。最后，s和e分别表示验证的最早（即1984年）和最晚年份（即2017年）。

使用bootstrap方法计算CNN和动力预测系统的预测技巧的置信区间。首先，我们随机选择N个集合成员。N是每个预测系统的集合成员数量（例如，对于CNN模型，N为40）。在随机选择过程中允许重复选择；选定的集合成员可以再次被选中。然后计算集合平均值的预测技巧。重复此过程10,000次：使用预测技巧的第250个最高值和最低值来定义95%置信区间。

**厄尔尼奥类型预测。** 使用Nino3指数（150∘−90∘W,5∘S−5∘N 区域平均SST异常）和Nino4指数（160∘E−150∘W,5∘S−5∘N 区域平均SST异常）定义EP型、CP型和混合型厄尔尼奥事件如下。UCEI表示统一复合ENSO指数。

其中

\[
\text{UCEI} = (N_3 + N_4) + (N_3 - N_4)\,i = r\,e^{i\theta},
\qquad
r = \sqrt{(N_3 + N_4)^2 + (N_3 - N_4)^2}
\]

\[
\theta =
\begin{cases}
\displaystyle
\arctan\!\left(\dfrac{N_3 + N_4}{N_3 - N_4}\right), & N_3+N_4>0 \\[6pt]
\displaystyle
\arctan\!\left(\dfrac{N_3 + N_4}{N_3 - N_4}\right) + \pi, & N_3+N_4<0,\;N_3-N_4>0 \\[6pt]
\displaystyle
\arctan\!\left(\dfrac{N_3 + N_4}{N_3 - N_4}\right) - \pi, & N_3+N_4<0,\;N_3-N_4<0
\end{cases}
\]

EP型厄尔尼奥：\(15^{\circ} < \theta < 90^{\circ}\)；  
CP型厄尔尼奥：\(-90^{\circ} < \theta < -15^{\circ}\)；  
混合型厄尔尼奥：\(-15^{\circ} < \theta < 15^{\circ}\)。

总的来说，当DJF季节的r值大于其标准差时，定义为厄尔尼奥事件。CMIP5输出中的厄尔尼奥事件使用相同的方法进行分类。用于训练CNN模型的样本数量为872。

基于随机预测对厄尔尼奥类型预测进行显著性检验。在随机分类厄尔尼奥事件后，计算随机预测的命中率。每种类型厄尔尼奥的发生率从CMIP5历史模拟中获得（即EP型、CP型和混合型厄尔尼奥分别为30%、26%和44%）。我们注意到，使用CMIP5档案中气候发生率的随机预测比使用相等发生率的随机预测具有更高的命中率。重复此过程10,000次。之后，使用命中率的第250个最高值和最低值来定义95%置信区间。

**用于ENSO预测的前馈神经网络模型。** 为了比较CNN模型中的ENSO预测技巧，我们基于前馈神经网络方法建立了一个非线性统计模型。使用SST和热含量的经验正交函数（EOF）主成分作为神经网络模型的预测因子。除了印太地区（40∘E−100∘W,20∘S−20∘N）的EOF主成分外，还分别获得了大西洋（100−0∘W,30∘S−30∘N）和北太平洋（120∘E−100∘W,20−60∘N）的EOF主成分，并将它们用作预测因子，以考虑大西洋和北太平洋气候变化对ENSO的远程影响。被预测量是Nino3.4指数。除了1871年至1973年的再分析数据外，我们还像本研究中的CNN一样，使用CMIP5模型的输出训练神经网络模型。CMIP5模型输出的主成分时间序列是通过计算模式投影系数到观测EOF特征向量上获得的。隐藏层数量为两层，每层隐藏神经元数量设置为20。激活函数为双曲正切函数。

进行了一系列不同数量预测因子的敏感性回溯预测实验，以预测DJF Nino3.4指数的18个月提前期（扩展数据表3）。最佳神经网络模型的预测技巧为0.52，使用印太地区9个EOF主成分以及大西洋和北太平洋7个EOF主成分作为预测因子，而CNN的预测技巧为0.64。神经网络模型的相关技巧随预测因子数量的微小变化而强烈变化。因此，在其他不同设置的神经网络模型中很难达到与最佳神经网络模型相似的相关技巧。通过最佳设置的神经网络模型，我们进行了回溯预测实验，以计算Nino3.4的全季节相关性，提前期长达24个月（扩展数据图1）。与Nino3.4的全季节相关技巧比较表明，CNN预测ENSO的能力系统性地优于神经网络模型。

**观测数据集和CMIP5输出。** 使用21个CMIP5模型产生的历史模拟数据训练CNN模型。扩展数据表1给出了运行模型的机构和积分周期的详细信息。使用所有CMIP5模型的一个集合成员训练CNN模型。因此，对于每个目标月，用于训练CNN模型的CMIP5样本总数为2,961。此外，还使用来自简单海洋数据同化版本2.2.4¹⁶的103年再分析数据（1871年至1973年之间）训练CNN模型。1974年之后的再分析产品未用于任何训练过程，以确保训练期和验证期相互独立。为了通过与观测值比较来验证模型的性能，从全球海洋数据同化系统（GODAS）再分析（1984-2017）³⁶收集了月平均SST和热含量数据，而925 hPa水平风矢量和降水数据从ERA-Interim档案（1984-2017）³⁷获得。使用了北美多模式集合第一阶段（1984-2017）³⁸和SINTEX-F（1984-2017）⁵'³⁹的八个模型来比较Nino3.4指数与CNN模型的预测性能。使用NMME进行了提前期长达12个月的预测，使用SINTEX-F进行了提前期长达24个月的预测。动力预测系统的异常是通过减去每个系统相对于预测提前期的气候平均值计算的。为了进行分析，空间分辨率插值为2.5∘×2.5∘，为了训练CNN模型，空间分辨率插值为5∘×5∘，以减少权重系数的数量。

**数据可用性**

与本文相关的数据可从以下网址下载：SODA版本2.2.4，[https://climatedataguide.ucar.edu/climate-data/soda-simple-ocean-data-assimilation](https://climatedataguide.ucar.edu/climate-data/soda-simple-ocean-data-assimilation)；GODAS，[https://www.esrl.noaa.gov/psd/data/gridded/data.godas.html](https://www.esrl.noaa.gov/psd/data/gridded/data.godas.html)；ERA-Interim，[https://apps.ecmwf.int/datasets/data/interim-full-daily](https://apps.ecmwf.int/datasets/data/interim-full-daily)；NMME第一阶段，[https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/)；以及CMIP5数据库，[https://esgf-node.llnl.gov/projects/cmip5/](https://esgf-node.llnl.gov/projects/cmip5/)。

**代码可用性**

使用了TensorFlow（[https://www.tensorflow.org](https://www.tensorflow.org)）库来构建使用CNN的统计预测模型。CNN模型的代码可从[https://doi.org/10.5281/zenodo.3244463下载](https://doi.org/10.5281/zenodo.3244463下载)。

**参考文献**

32. Goodfellow, I., Bengio, Y., Courville, A. & Bengio, Y. Deep Learning (MIT Press, 2016).  
33. Yoo, J. H. & Kang, I. S. Theoretical examination of a multi-model composite for seasonal prediction. Geophys. Res. Lett. 32, L18707 (2005).  
34. Kalchbrenner, N., Grefenstette, E. & Blunsom, P. A convolutional neural network for modelling sentences. In Proc. 52nd Ann. Meet. Association for Computational Linguistics 655-665 (Association for Computational Linguistics, 2014).  
35. Wu, A., Hsieh, W. W. & Tang, B. Neural network forecasts of the tropical Pacific sea surface temperatures. Neural Netw. 19, 145-154 (2006).  
36. Behringer, D. W. & Xue, Y. Evaluation of the global ocean data assimilation system at NCEP: The Pacific Ocean. In Proc. Eighth Symp. on Integrated Observing and Assimilation Systems for Atmosphere, Oceans, and Land Surface (AMS 84th Annual Meeting) (AMS, 2004).  
37. Dee, D. P. et al. The ERA-Interim reanalysis: configuration and performance of the data assimilation system. Q. J. R. Meteorol. Soc. 137, 553-597 (2011).  
38. Kirtman, B. P. et al. The North American multimodel ensemble: phase-1 seasonal-to-interannual prediction; phase-2 toward developing intraseasonal prediction. Bull. Am. Meteorol. Soc. 95, 585-601 (2014).  
39. Luo, J.-J., Liu, G., Hendon, H., Alves, O. & Yamagata, T. Inter-basin sources for two-year predictability of the multi-year La Niña event in 2010-2012. Sci. Rep. 7, 2276 (2017).

**致谢** 本研究由韩国气象管理局研发计划KMI2018-03214资助。Y.-G.H.由韩国国家研究基金会（NRF）资助的基础科学研究计划（NRF-2016R1A6A1A03012647）支持。J.-J.L.由南京信息工程大学"引进人才启动基金"支持。感谢W. Merryfield提供的评论以及T. Doi提供的部分用于验证的SINTEX-F回溯预测数据。

**作者贡献** Y.-G.H.和J.-H.K.设计了实验并进行了分析。Y.-G.H.撰写了大部分手稿。J.-H.K.和Y.-G.H.进行了CNN回溯预测实验。J.-J.L.进行了SINTEX-F回溯预测实验并报告了结果。所有作者讨论了研究结果并审阅了手稿。

**竞争利益** 作者声明没有竞争利益。

**附加信息**

补充信息可在以下网址获取：https://doi.org/10.1038/s41586-019-1559-7。  
通讯和材料请求应发送至Y.-G.H.。  
重印和许可信息可在http://www.nature.com/reprints获取。  
**扩展数据图和表格（仅翻译标题和简要说明，图像内容无法直接嵌入）**

**扩展数据图1 | CNN模型与前馈神经网络模型ENSO相关技巧的比较。** a，CNN模型（红色）和前馈神经网络模型（蓝色）中三年移动平均Nino3.4指数作为预测提前期函数的全季节相关技巧。验证期为1984年至2017年。b，c，CNN模型（b）和前馈神经网络模型（c）中针对每个日历月的Nino3.4指数相关技巧。阴影区域表示相关技巧超过0.5的预测。

**扩展数据图2 | CMIP5数据集和迁移学习对CNN模型技巧的改进。** a，不同数量CMIP5样本下Nino3.4指数18个月提前期的全季节相关技巧。红色区域表示可用观测样本的数量。我们注意到，迁移学习未应用于一系列敏感性测试（即训练期内的观测数据未用于构建CNN模型）。b，使用和不使用迁移学习的Nino3.4全季节相关技巧作为预测提前期函数的图。不使用迁移学习的CNN模型是在单个训练期内使用训练期内所有CMIP5和观测样本（即1871年至1973年）构建的。因此，不使用迁移学习的CNN模型的样本数量与使用迁移学习的模型完全相同。

**扩展数据图3 | 气候指数的时间序列。** a-d，SON季节（a）的IOD指数（50−70∘E, 10∘S−10∘N 区域平均SST与90−110∘E, 15∘−0∘S 区域平均SST之差），JFM季节（b）的印度洋盆地范围增暖（IOBW）指数（40−110∘E, 15∘S−10∘N 区域平均SST），MJJ季节（c）的西半球暖池（WHWP）指数（60−105∘E, 10−35∘N 区域平均SST），以及DJF季节（d）的太平洋经向模（PMM）指数（使用SST和10米风在175∘E−95∘W,21∘S−32∘N 区域进行第一主成分分析（MCA）获得）。1997/98年厄尔尼奥事件之前的值，正值用红色星号表示，负值用蓝色星号表示。

**扩展数据图4 | 1997/98年厄尔尼奥事件的时间演变。** a，MJJ 1996；b，ASO 1996；c，NDJ 1996；和d，FMA 1997的SST（阴影）和850 hPa风矢量（矢量）。全球地图由Matplotlib³¹生成。

**扩展数据图5 | 厄尔尼奥事件的区域平均热力图值。** a，b，EP型厄尔尼奥事件（a）和CP型厄尔尼奥事件（b）在所有厄尔尼奥事件中五个海洋区域（即南太平洋、赤道太平洋、北太平洋、印度洋和赤道大西洋）的区域平均热力图。这些区域定义为：南太平洋，\[160∘E−60∘W,57.5∘−17.5∘S\]；赤道太平洋，\[120∘E−80∘W,17.5∘S−22.5∘N\]；北太平洋，\[120∘E−100∘W,22.5−62.5∘N\]；印度洋，\[40∘−120∘E,37.5∘S−22.5∘N\]；和赤道大西洋，\[60∘−0∘W,17.5∘S−22.5∘N\]。水平虚线表示五个海洋盆地中显示的厄尔尼奥事件热力图值的一个标准差。我们注意到，只分析了CNN正确预测类型的厄尔尼奥事件的热力图。

**扩展数据图6 | CP型厄尔尼奥前兆引发的SST模式。** a，b，NDJ季节印度洋（a）和南太平洋（b）CP型厄尔尼奥前兆的模式回归指数对12个月提前期预测的SST异常进行回归。CP型厄尔尼奥前兆的模式回归指数是通过计算NDJ SST和热含量异常对图4f中CP型厄尔尼奥事件选定异常的模式回归获得的。黑框表示计算模式回归指数的区域。我们注意到，两个回归的SST模式都被分类为CP型厄尔尼奥²⁴。全球地图由Matplotlib³¹生成。

**扩展数据表1 | CMIP5模型的详细信息。** 用于训练CNN模型的CMIP5模型列表。

**扩展数据表2 | 用于训练和验证CNN模型的数据集。** 用于训练和验证CNN模型的数据集和周期。我们注意到，CMIP5模型中的年份完全取决于预设的温室气体强迫，因此未向CMIP5历史模拟中添加任何观测信息。

**扩展数据表3 | 不同数量预测因子下前馈神经网络模型的技巧。** 不同数量EOF主成分（PCs）作为预测因子下，DJF季节Nino3.4指数18个月提前期的前馈神经网络模型的相关技巧。验证期为1984年至2017年。CNN针对DJF Nino3.4指数18个月提前期的相关技巧为0.64。

**扩展数据表4 | 厄尔尼奥类型12个月提前期预测结果。** CNN和动力预测系统中，DJF季节EP型、CP型和混合型厄尔尼奥事件12个月提前期预测结果。底部一行表示验证期（1984年至2017年）的命中率。括号中的值表示CNN从1976年至2017年的命中率。绿色阴影表示预测正确。