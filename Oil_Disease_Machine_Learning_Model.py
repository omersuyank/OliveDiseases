import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Veri yüklemesi C:\Users\remou\Desktop\MakineOgrenmesiTarim\input
folder = "C:/Users/remou/Desktop/MakineOgrenmesiTarim/input/"
sub_folders = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
print(sub_folders)
files = []
labels = []

for sub_folder in sub_folders:
    sub_files = [os.path.join(sub_folder, file) for file in os.listdir(sub_folder) if file.endswith('.jpg')]
    files.extend(sub_files)
    labels.extend([os.path.basename(sub_folder)] * len(sub_files))

data = pd.DataFrame({'files': files, 'Labels': labels})

# Veriyi ekrana yazdır
print(data.head())

# Veri ön işleme ve bölme
X = []
for file in data['files']:
    img = cv2.imread(file)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'yi RGB'ye dönüştür
    img = img.astype('float32') / 255.0  # Normalizasyon
    X.append(img)

y = LabelEncoder().fit_transform(data['Labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# VGG16 ile özellik çıkarma
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
X_train_features = base_model.predict(np.array(X_train))
X_test_features = base_model.predict(np.array(X_test))


# RandomForestClassifier modeli ile eğitim ve tahmin
rf_model = RandomForestClassifier(n_estimators=190, random_state=25)
rf_model.fit(X_train_features.reshape(X_train_features.shape[0], -1), y_train)
rf_y_pred = rf_model.predict(X_test_features.reshape(X_test_features.shape[0], -1))


# DecisionTreeClassifier modeli ile eğitim ve tahmin
dt_model = DecisionTreeClassifier(random_state=25)
dt_model.fit(X_train_features.reshape(X_train_features.shape[0], -1), y_train)
dt_y_pred = dt_model.predict(X_test_features.reshape(X_test_features.shape[0], -1))

# SVC modeli ile eğitim ve tahmin
svc_model = SVC(random_state=25)
svc_model.fit(X_train_features.reshape(X_train_features.shape[0], -1), y_train)
svc_y_pred = svc_model.predict(X_test_features.reshape(X_test_features.shape[0], -1))

# KNeighborsClassifier modeli ile eğitim ve tahmin
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_features.reshape(X_train_features.shape[0], -1), y_train)
knn_y_pred = knn_model.predict(X_test_features.reshape(X_test_features.shape[0], -1))

# Doğrulukları saklamak için bir sözlük oluştur
accuracy_scores = {
    "Random Forest": accuracy_score(y_test, rf_y_pred),
    "Decision Tree": accuracy_score(y_test, dt_y_pred),
    "Support Vector": accuracy_score(y_test, svc_y_pred),
    "K Neighbors": accuracy_score(y_test, knn_y_pred)
}

# Ortalama, min ve max doğruluk
average_accuracy = np.mean(list(accuracy_scores.values()))
min_accuracy = min(accuracy_scores.values())
max_accuracy = max(accuracy_scores.values())
best_model = max(accuracy_scores, key=accuracy_scores.get)

# Çubuk grafik oluştur
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title("Makine Öğrenimi Algoritmalarının Doğrulukları")
plt.xlabel("Algoritmalar")
plt.ylabel("Doğruluk")
plt.ylim(0, 1)  # Y eksenini 0-1 arasında sınırla
plt.xticks(rotation=45)  # X eksenindeki yazıları 45 derece döndür
plt.axhline(average_accuracy, color='red', linestyle='--', label='Ortalama Doğruluk: {:.2f}'.format(average_accuracy))
plt.text(3.5, average_accuracy + 0.01, 'Ortalama Doğruluk: {:.2f}'.format(average_accuracy), ha='center', va='bottom', color='red')
plt.axhline(min_accuracy, color='blue', linestyle='--', label='Min Doğruluk: {:.2f}'.format(min_accuracy))
plt.text(3.5, min_accuracy - 0.02, 'Min Doğruluk: {:.2f}'.format(min_accuracy), ha='center', va='bottom', color='blue')
plt.axhline(max_accuracy, color='green', linestyle='--', label='Max Doğruluk: {:.2f}'.format(max_accuracy))
plt.text(3.5, max_accuracy - 0.02, 'Max Doğruluk: {:.2f}'.format(max_accuracy), ha='center', va='bottom', color='green')
plt.legend()
plt.tight_layout()
plt.show()

# En iyi modeli belirt
print("En iyi model:", best_model)




# Performans metrikleri
print("Random Forest Classifier:")
print(classification_report(y_test, rf_y_pred))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))

print("Decision Tree Classifier:")
print(classification_report(y_test, dt_y_pred))
print("Accuracy:", accuracy_score(y_test, dt_y_pred))

print("Support Vector Classifier:")
print(classification_report(y_test, svc_y_pred))
print("Accuracy:", accuracy_score(y_test, svc_y_pred))

print("K Neighbors Classifier:")
print(classification_report(y_test, knn_y_pred))
print("Accuracy:", accuracy_score(y_test, knn_y_pred))

import joblib #modelleri kaydediyoruz

# RandomForestClassifier modelini kaydetmek
joblib.dump(rf_model, 'Oil_rf_model.pkl')

# DecisionTreeClassifier modelini kaydetmek
joblib.dump(dt_model, 'Oil_dt_model.pkl')

# SVC modelini kaydetmek
joblib.dump(svc_model, 'Oil_svc_model.pkl')

# KNeighborsClassifier modelini kaydetmek
joblib.dump(knn_model, 'Oil_knn_model.pkl')
