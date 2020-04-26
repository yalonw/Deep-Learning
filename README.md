###### Date: 2020.04.20 

# 前情提要
使用 `TensorFlow` 時，可以選擇安裝或不安裝：）

## 本地運行
- 系統環境
  - 64-bit systems
  - Windows 7 or later 
  - macOS 10.12.6 (Sierra) or later (no GPU support)
- 安裝軟體
  - python 3.7
  - [TensorFlow 2](https://www.tensorflow.org/install)：`pip install tensorflow`

## 雲端運行
- [Google Colab / Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
  - Colab 是由 Google 提供以幫助傳播機器學習教育和研究。  
  - Colab 是一個 Jupyter notebook 環境，不需要安裝即可使用，且完全在雲端中運行。

</br></br>

# AI 基礎概念
### **機器學習**（Machine Learning）
  - 傳統統計機率 
  - 擅長處理表格（結構化資料）  
  - 資料量需要「千」筆以上，也考慮問題難度，問題若簡單則資料量可不用太多
  - ML 結構：資料 → 特徵擷取 (基於人類知識) → 模型 → 答案
  - 詳見：[https://github.com/yalonw/Machine_Learning](https://github.com/yalonw/Machine_Learning#ml-model-%E5%BB%BA%E7%AB%8B%E6%AD%A5%E9%A9%9F)

### **深度學習**（Deep Learning）
  - 是模仿人類大腦的認知過程，故又稱為**神經網路**
  - 是透過「**組合＋遺忘**」得到結果，而傳統統計無法做到「組合」
  - 擅長處理**圖片、文字、語言**等抽象型資料（非結構化資料）
  - 資料量需要「萬」筆以上，也考慮問題難度，問題困越難資料量需要越多
  - DL 結構：資料 → ~~特徵擷取~~ (由模型自學) → 模型 → 答案

</br></br>

# DL 神經網路架構
### **1. Perceptron (單層)感知器**
  - 是一種「**線性**分類器」
  - 只有一個神經元 / unit / score 
  - 分類方法為「All-or-none law 全有全無律」；只有二種選擇 output = 0 or 1
  - 缺點：只用一條線做二分法，無法解決現實問題

![](./pic/01-Perceptron.png)

--------
  - 問題：二分法（output = 0 or 1）太武斷！
  - 解法：透過 **`sigmoid` 激活函數** 轉換（或稱 Logistic Regression）
  - 成果：使預測結果轉為**機率**（output = 0 ~ 1）的形式，  
    　　　而轉換後並不會改變線性分類的結果  

![](./pic/02-Logistic_Regression.png)

--------
  - 問題：線性分類不符合現實問題！例如：XOR problem
  - 解法：透過**多個**神經元 / unit / score 組合  
  - 成果：雖然每個神經元組合仍為線性分類，但中間經過 `sigmoid` 轉換，  
    　　　因此最後 output 是帶有非線性的元素

![](./pic/03-Non-linear_classifier.png)

--------
</br>

### **2. Multilayer Perceptron 多層感知器（MLP）**
  - 是一種「**非線性**分類器」
  - 有多條神經，多個神經元 / unit / score
  - 透過「線性組合＋非線性轉換」，達到「非線性分類」
  - 範例：[MLP_1_MNIST](https://github.com/yalonw/Deep_Learning/blob/master/MLP_1_MNIST.ipynb)、[MLP_2_Fashion_MNIST](https://github.com/yalonw/Deep_Learning/blob/master/MLP_2_Fashion_MNIST.ipynb)

![](./pic/04-Multilayer_Perceptron.png)

--------
  - 問題：「權重（w）」如何選擇？
  - 解法：「**隨機梯度下降**（stochastic gradient descent）」
    1. 調整方式：隨意選擇權重，再逐步調整，直到誤差最小（min(Loss)），即為最佳預測結果（predict）  
    2. 調整方向：「**負斜率**（negative slope） 或 **梯度下降**（gradient descent）」的方向
       - 二維空間：- dy/dx　　　　　　     --> 負斜率  
       - 三維空間：- ( ∂x/∂z, ∂y/∂z )　　 --> 梯度下降
       - 多維空間：- ( ∂Loss/∂w1, ∂Loss/∂w2, ..., ∂Loss/∂wn )
  - 數學表示式：
    ![](./pic/04-stochastic_gradient_descent.png)

![](./pic/04-MLP_stochastic_gradient_descent.png)

> **誤差反向傳播**（Backpropagation）：  
  是對神經網絡中所有權重（w）計算損失函數（Loss）的梯度：∂Loss/∂w，  
  並結合隨機梯度下降（stochastic gradient descent）更新權值，以最小化損失函數。

--------
  - 問題：「梯度消失（Vanishing Gradient problem）」  
    　　　 因 `sigmoid` 激活函數的微分介於 0~0.25 之間，經過多層神經網路後，  
    　　　 會使較遠的權值沒有被校正正確，權值會越調整越差，甚至是不調整

![](./pic/05-vanishing_gradient_problem.png)

  - 解法：將中間層激活函數從 **`sigmoid`** 改為 **`relu`**  
  - 成果：`relu` 被激活後，微分 = 1；因此在誤差反向傳播時，可以正確傳遞  

![](./pic/06-sigmoid.png)
![](./pic/06-relu.png)

--------
  - 問題：「激活函數（Activation Function）」如何選擇？
  - **中間層** 激活函數：
      - 必定選擇 `relu`，來將「特徵組合」從線性轉成非線性分類   
      - 而不選擇 `sigmoid` 是因為經過 `sigmoid` 轉換後，  
        誤差（∂Loss/∂w）會隨著傳播越遠越被消弭，而產生「梯度消失問題」
  + **最終層** 激活函數：  
    根據需要的 output 形式來選擇，例如
      - 二元分類：選擇 `sigmoid`，讓機率介於 0 ~ 1 之間
      - 二元分類：選擇 `tanh`　　，讓機率介於 -1 ~ 1 之間
      - 多元分類：選擇 `softmax`，讓機率相加等於 1
      - 回　　歸：不需要激活函數

![](./pic/06-relu.png)
![](./pic/07-relu_and_softmax.png)

> 線性分類器與非線性分類器的區別：  
  圖左：非線性分類器  
  圖右：線性分類器，又稱為一刀切
  ![](https://upload.wikimedia.org/wikipedia/commons/f/fe/Kernel_Machine.svg)

--------
</br>

### **3. Convolutional Neural Network 卷積神經網路（CNN）**
- 在做「影像」辨識時，  
  如果只用 MLP 做出基於**像素 (pixel)** 的分類預測，這樣是不行的！  
  所以有了 CNN ，先對影像做「**特徵萃取**」，再用 MLP 做出基於**特徵**的分類預測～

- **CNN = Convolution + Pooling + Flatten + MLP**
  1. Convolution 卷積：  
     目的：對影像進行**特徵萃取/過濾**，以得到有用的(想要的)資訊  
     方法：改變影像通道數 (channel)   
     注意：(1) 輸入的資料維度必須有「通道數」， shape = ( batch, height, width, **channel** )  
     　　　(2) 圖片越大，才能做越多層卷積 

     ![](./pic/11-convolution.png)

     zero padding：為了不改變影像尺寸，在影像外圍加白邊
     
     ![](./pic/12-convolution+padding.png)

  2. Pooling 池化：  
     目的：減少影像資料量與計算量，並保留重要資訊  
     方法：改變影像高寬 (height, width)  
     例如：MaxPooling、AveragePooling

     ![](./pic/13-polling.png)

  3. Flatten：將影像轉換成一維陣列 (1D Array)

- CNN 卷積神經網路的架構 = **Convolution + Pooling + Flatten + MLP** 

  ![](./pic/14-convolution+padding+pooling.png)
  ![](./pic/15-cnn.png)

- 範例：[CNN_simple_CIFAR10](https://github.com/yalonw/Deep_Learning/blob/master/CNN_simple_CIFAR10.ipynb)

---
- 問題：如何選擇濾鏡 = filter = kernel map？  
  解法：模型會自己解決 XD

- 問題：卷積需要做幾層 layer？每層卷積需要幾個 filter？  
  解法：參考大大的模型 XD  
  例如：**ImageNet** 比賽中，獲獎的演算法～  
  　　　(1) AlexNet：紀念版模型  
  　　　(2) VGG 系列：經典版模型  
  　　　(3) ResNet 系列：特色是使用更多層 layer  
  　　　(4) Inception 系列：特色是使用多種不同大小的 Kernel map (filter)  

  ![參考來源：https://towardsdatascience.com/neural-network-architectures-156e5bad51ba](https://miro.medium.com/max/1400/1*kfpO_fJ4bc92sffY4bxnSA.jpeg)
  > [**ImageNet**](http://www.image-net.org) 是 Hinton 和他的學生建立的大型圖像資料庫。  
    上圖中，縱軸的 Top-1 Accuracy，表示只預測一次且正確的機率。
  

</br></br>

# 範例程式
> 請按順序學習，每個範例中都有更深入的解釋 :blush:
1. MLP-1：[MNIST](https://github.com/yalonw/Deep_Learning/blob/master/MLP_1_MNIST.ipynb)
2. MLP-2：[Fashion_MNIST](https://github.com/yalonw/Deep_Learning/blob/master/MLP_2_Fashion_MNIST.ipynb)
3. CNN-simple：[CIFAR10](https://github.com/yalonw/Deep_Learning/blob/master/CNN_simple_CIFAR10.ipynb)