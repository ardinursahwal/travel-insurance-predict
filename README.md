# Laporan Proyek Machine Learning

### Nama : Ardi Nursahwal

### Nim : 211351022

### Kelas : Malam B

## Domain Proyek

Web App ini dirancang untuk membantu dalam memprediksi pelanggan untuk membeli asuransi perjalanan atau tidak berdasarkan. Namun, penting untuk dicatat bahwa hasil prediksi dari Web App ini bukanlah jaminan pasti, melainkan merupakan indikasi awal yang disediakan berdasarkan data yang ada.

## Business Understanding

Tujuan utama dari Web App ini adalah untuk memberikan informasi yang dapat membantu dalam memahami potensi keinginan pelanggan untuk membeli asuransi perjalanan. Dengan informasi ini, perusahaan asuransi dapat meningkatkan strategi pemasaran dan pendekatan kepada pelanggan potensial, serta menyediakan layanan yang lebih sesuai dengan kebutuhan mereka.

### Problem Statements

Ketidaktahuan tentang preferensi pelanggan terkait asuransi perjalanan sering kali menjadi hambatan dalam upaya pemasaran yang efektif. Perusahaan asuransi membutuhkan pemahaman yang lebih baik mengenai perilaku pembelian pelanggan untuk meningkatkan penjualan.

### Goals

-   Menggunakan data historis dan faktor-faktor terkait untuk memprediksi kemungkinan pembelian asuransi perjalanan oleh pelanggan.
-   Memberikan hasil prediksi yang dapat membantu perusahaan asuransi dalam menyusun strategi pemasaran yang lebih efektif dan menyesuaikan penawaran layanan sesuai dengan preferensi pelanggan potensial.

## Data Understanding

Dataset yang digunakan berasal dari sebuah perusahaan perjalanan yang ingin memprediksi minat pelanggan terhadap paket asuransi perjalanan, berdasarkan riwayat database mereka. Data ini mencakup hampir 2000 pelanggan sebelumnya dan tujuannya adalah untuk membangun model yang dapat memprediksi minat pelanggan dalam membeli paket asuransi perjalanan berdasarkan parameter tertentu yang tersedia dari riwayat pembelian sebelumnya.
Link dataset [Travel insurance Prediction](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data).

### Variabel-variabel pada chronic kidney disease adalah sebagai berikut:

-   **Age**: Usia dari pelanggan. (tipe data int64)
-   Employment **Type**: Sektor di mana pelanggan bekerja. (tipe data int64)
-   **GraduateOrNot**: WhetherApakah pelanggan merupakan lulusan perguruan tinggi atau tidak. (tipe data object)
-   **AnnualIncome**: Pendapatan tahunan pelanggan dalam Rupee India (dibulatkan menjadi 50 ribu Rupee terdekat). (tipe data int64)
-   **FamilyMembers**: Jumlah anggota dalam keluarga pelanggan. (tipe data int64)
-   **ChronicDisease**: Apakah pelanggan menderita penyakit atau kondisi serius seperti diabetes, tekanan darah tinggi, atau asma, dll. (tipe data int64)
-   **FrequentFlyer**: Data turunan berdasarkan riwayat pelanggan dalam memesan tiket pesawat minimal 4 kali dalam 2 tahun terakhir [2017-2019]. (tipe data object)
-   **EverTravelledAbroad**: Apakah pelanggan pernah melakukan perjalanan ke luar negeri (tidak harus menggunakan layanan perusahaan). (tipe data object)
-   **TravelInsurance**: Apakah pelanggan membeli paket asuransi perjalanan selama penawaran perkenalan yang diadakan pada tahun 2019. (tipe data int64)

## Data Preparation

### Unduh datasets

Pertama upload file kaggle agar bisa mengunduh datasetnya

```python
from google.colab import files
files.upload()
```

Lalu membuat folder untuk menyimpan filenya

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Dan mendownload datasetnya

```python
!kaggle datasets download -d tejashvi14/travel-insurance-prediction-data
```

Serta mengekstrak file yang telah diunduh ke dalam folder tersebut.

```python
!unzip travel-insurance-prediction-data.zip -d travel-insurance-prediction-data
!ls travel-insurance-prediction-data
```

### Import libary yang diperlukan

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn import metrics


import pickle
import warnings
warnings.simplefilter("ignore")
```

### Data discovery

Pertama kita buat dataframe dari dataset yang telah kita download diatas

```python
df = pd.read_csv('/content/travel-insurance-prediction-data/TravelInsurancePrediction.csv')
```

Cek 5 data teratas pada dataset, sepertinya kolom Unnamed tidak diperlukan tetapi abaikan dulu untuk sekarang

```python
df.head()
```

Dataset yang kita gunakan memiliki baris yang cukup banyak yaitu 1987 baris data dengan 10 kolom

```python
df.shape
```

Mari kita lihat tipe data pada masing-masing kolom, hmm disini kita memiliki total 4 kolom data object kita akan memperbaikinya pada bagian preprocessing

```python
df.info()
```

Cek data null pada tiap kolom, dan mantap tidak ada data yang null

```python
df.isnull().sum()
```

### EDA

Pertama kita lihat korelasi antar kolom pada data kita

```python
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=True);
```

![heatmap](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/77a53ece-6e44-48b1-9ca8-6281fc2e13fc)

Menampilkan distribusi pendapatan tahunan dan rata-rata pendapatan pada judul plot

```python
plt.style.use("classic")
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(df["AnnualIncome"], color="g")
plt.title(f"Annual Income Distribution [ \u03BC: {df['AnnualIncome'].mean():.2f} ]")
plt.show()
```

![income](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/b390882b-89a5-4366-a55f-331b57a50161)

Dapat kita lihat sebagian besar anggota keluarga 4 dengan jumlah terendah 8 dan 9.

```python
sns.countplot(x='FamilyMembers',data=df)
plt.show()
```

![family](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/c1fef503-ab18-467d-a89e-5ec735c6350e)

Umur 28 paling banyak dan 35 paling rendah tetapi 34 paling banyak kedua setelah 28

```python
sns.countplot(x='Age',data=df)
plt.show()
```

![age](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/ca009b75-d907-47ca-a46f-018f675f21de)

Menampilkan pelanggan yang pernah melakukan perjalanan ke luar negeri dan tidak, sebanyak 72,2% tidak pernah melakukan perjalanan ke luar negeri

```python
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(8,6))
plt.pie(x=df["ChronicDiseases"].value_counts(),
        colors=["blue","darkorchid"],
        labels=["Bukan Abroad Travellers","Abroad Travellers"],
        shadow = True,
        explode = (0, 0.1),
        autopct='%1.1f%%'
        )
plt.show()
```

![aboard](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/c7e0b45e-848e-469b-99cc-15431a1d7978)


Dan menampilkan pelanggan yang pernah memiliki ansuransi perjalanan dan tidak, sebanyak 64,3% tidak pernah melakukan perjalanan ke luar negeri

```python
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(8,6))
plt.pie(x=df["TravelInsurance"].value_counts(),
        colors=["mediumorchid","orange"],
        labels=["Tidak memiliki Travel Insurance","Memilki Travel Insurance"],
        shadow = True,
        explode = (0, 0.1),
        autopct='%1.1f%%'
        )
plt.show()
```

![travel](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/c0f07b70-8218-446a-bf22-c21c715e7e73)

### Preprocessing

Mari kita hapus kolom yang tidak diperlukan

```python
df.drop(["Unnamed: 0"], axis=1, inplace=True)
```

Ubah data object menjadi numerikal

```python
df['GraduateOrNot'] = df['GraduateOrNot'].map({'Yes': 1, 'No': 0})
df['FrequentFlyer'] = df['FrequentFlyer'].map({'Yes': 1, 'No': 0})
df['EverTravelledAbroad'] = df['EverTravelledAbroad'].map({'Yes': 1, 'No': 0})

# Government Sector : 1, Private Sector/Self Employed : 0
df["Employment Type"] = df["Employment Type"].map({"Government Sector" : 1, "Private Sector/Self Employed" : 0})
```

Setalah proses preprocessing selanjutnya kita akan ke proses modeling

## Modeling

Tentu tahap pertama modeling yang akan kita lakukan adalah membuat variable label dan feature

```python
x = df.drop(['TravelInsurance'],axis=True)
y = df['TravelInsurance']
```

Lalu split data menjadi dua subset yaitu untuk training 80% dan testing 20%

```python
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
```

Normalisasi data feature

```python
mscale=MinMaxScaler()
mscale.fit_transform(x_train)
mscale.transform(x_test)
```

Dapat kita lihat perbedaan akurasi antara test dan train memiliki perbedaan yang cukup signifikan, ini bisa menjadi indikasi overfitting pada model

```pyton
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predict = model.predict(x_test)

train_acc = model.score(x_train,y_train)
test_acc = model.score(x_test,y_test)
print('Training acc:',train_acc,'Test acc:',test_acc)
```

```bash
Training acc: 0.920075519194462 Test acc: 0.7738693467336684
```

Dan kita akan coba model prediksi sebelum melakukan pruning

```python
# data = np.array([[31,0,1,400000,6,1,0,0]]) # 0
data = np.array([[26,1,1,1400000,5,0,1,1]]) # 1

prediction = model.predict(data)
prediction[0]
```

### Visuliasai hasil algoritma

Dapat kita lihat D-Tree kita overfitting maka dari itu kita akan melakukan pendekatan pruning

```python
ind_col = [col for col in df.columns if col != 'TravelInsurance']
dep_col = 'TravelInsurance'

fig = plt.figure(figsize=(25,30))
_ = tree.plot_tree(model,
                   feature_names=ind_col,
                   class_names=dep_col,
                   filled=True)
```

![treeoverfit](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/cefc175d-1e19-4c08-9625-d7e2163548be)

Selanjutnya kita akan melakukan proses pruning atau mengidentifikasikan dan membuang cabang yang tidak diperlukan pada pohon yang telah terbentuk.
Disini saya menggunakan GridSearchCV untuk membantu dalam mencari kombinasi parameter terbaik dengan cara mencoba semua kombinasi yang mungkin yang telah ditentukan sebelumnya.

```python
model_prun = gcv.best_estimator_
model_prun.fit(x_train,y_train)

y_train_pred = model_prun.predict(x_train)
y_test_pred = model_prun.predict(x_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
```

```bash
Train score 0.8382630585273757
Test score 0.8190954773869347
```

Dan perbedaan sudah tidak terlalu signifikan hanya sekitar 2%

Selanjutnya melihat visualisasi tree sesudah pre prunning

```python
plt.figure(figsize=(10,10))
features = x.columns
classes = ['0','1']
tree.plot_tree(model_prun,feature_names=features,class_names=classes,filled=True)
plt.title('Sesudah Pre Pruning')
plt.show()
```

![treepreprun](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/b6d8903f-eb6d-43ae-9a89-18cc191e7150)

Selanjutnya kita akan lakukan post prunning, pertama menghitung nilai alpha pruning dan impurities pada lintasan pruning berdasarkan kompleksitas dari pohon keputusan pada data train.

```python
path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
```

Mencetak jumlah node dalam pohon keputusan terakhir yang dibuat dari nilai alpha pruning terakhir dalam daftar nilai alpha

```python
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)
print("Angka node dalam pohon terakhir adalah: {} dengan ccp_alpha: {}".format(
    clfs[-1].tree_.node_count, ccp_alphas[-1]))
```

Selanjutnya membuat sebuah Decision Tree Classifier dengan alpha pruning setara 0.013

```python
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.013)
clf.fit(x_train,y_train)
```

Lalu melihat akurasi yang didapatkan

```python
pred = clf.predict(x_test)
print("Training Accuracy :", clf.score(x_train, y_train))
print("Testing Accuracy :", accuracy_score(y_test,pred))
```

```bash
Training Accuracy : 0.8332284455632474
Testing Accuracy : 0.8241206030150754
```

Dapat kita lihat perbedaan akurasinya sekarang hanya sekitar 1%

Melihat decision tree setelah pruning dilakukan

```python
plt.figure(figsize=(10,10))
features = x.columns
classes = ['0','1']
tree.plot_tree(clf,feature_names=features,class_names=classes,filled=True)
plt.title('Setelah Pruning')
plt.show()
```

![treepostprun](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/e2f4e384-feea-4617-b023-7034afd0e4f4)

Berdasarkan decision tree di atas, dapat disimpulkan bahwa Annual Income adalah variabel yang paling berpengaruh. Semakin tinggi Annual Income, maka semakin besar kemungkinan memiliki travel insurance.

### Save model (pickle)

```python
filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))
```

## Evaluation

Untuk matrix evalusinya disini saya menggunakan confusion matrix, dengan confusion matrix kita dapat melihat perbandingan nilai prediksi dan aktual.

```python
y_pred = clf.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="cividis" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```

![matrix pn](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/c2114577-57ff-4c8b-9963-181cf6fc7d40)

Dari hasil confusion matrixnya dapat kita lihat 
Ada 251 prediksi yang benar bahwa bukan kelas positif (True Negative).
Terdapat 1 prediksi yang salah bahwa itu kelas positif (False Positive).
Ada 63 prediksi yang salah bahwa itu bukan kelas positif (False Negative).
Terdapat 77 prediksi yang benar bahwa itu kelas positif (True Positive).

## Deployment

[Travel insurance predict](https://travel-insurance-predict.streamlit.app/)

![streamlit](https://github.com/ardinursahwal/travel-insurance-predict/assets/148542995/c994a356-db45-4d66-b065-2854a03f0017)
