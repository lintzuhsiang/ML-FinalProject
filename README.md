# ML-FinalProject

Cyber Security Attack Defender
==============================

Team Work
==============================
分工:
  * 林子翔:進階 model 的設計 (second baseline)、deep learning layer 與 node 的測試及討論。
  * 董皓文:資料的前處理、特徵值的取出與轉換 (feature engineering)、初 始的 model 設計 (first baseline)。
  * 謝秉翰:初始 model 設計的改良、random forest 跟 support vector machine 的討論。
 
Dataset
==============================
Based on DARPA’98 IDS evaluation program
------------------------------------------
* About 5 million connection records
* Number of columns: 42
* 41 for connection information

Four types of attack
--------------------
• DOS: denial-of-service
• R2L: unauthorized access from a remote machine
• U2R: unauthorized access to local superuser (root) privileges
• probing: surveillance and other probing


Model
====================
Support Vector Machine Classification (SVC)
-------------------------------------------
We use scikit-learn Python package with three kernel: radial basis function (RBF)、linear、sigmoid.


Random Forest Classification (RF)
-------------------------------------------
We use Scikit-learn Python package RandomForestClassifier object.


Deep Learning
-------------------------------------------
四個 hidden layer。
• 四個 hidden layer 的 node 數量分別為 500、800、250 跟 100。
• 每一個 hidden layer 都使用 relu 的 activation function。
• 最後一層的分類使用 sigmoid function。
• 使用的 loss function 是 binary cross-entropy。
• 利用 adam(momentum + scaling)的方式更新 gradient descent。
• 為了避免 model 的 overfitting，會在經過每一層 layer 設定 dropout 一
半的 input。
• nb_epoch=10、batch_size=5000。



More Detail
===========
See Report.pdf
