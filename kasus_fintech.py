# Mengimpor Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengimpor Dataset
dataku = pd.read_csv("data_fintech.csv")

# Ringkasan Data
ringkasan = dataku.describe()
tipe_data = dataku.dtypes

# Revisi Kolom num_screens
dataku['screen_list'] = dataku.screen_list.astype(str) + ','
dataku['num_screens'] = dataku.screen_list.str.count(',')
dataku.drop(columns=['numscreens'], inplace=True)

# Cek Kolom hour
dataku.hour[1]
dataku.hour = dataku.hour.str.slice(1,3).astype(int)

# Mendefinisikan Variabel Khusus Numerik
dataku_numerik = dataku.drop(columns=['user','first_open','screen_list','enrolled_date'], inplace=False)

# Membuat Histogram
sns.set()
plt.suptitle('Histogram Data Numerik')
for i in range(0, dataku_numerik.shape[1]):
    plt.subplot(3,3,i+1)
    figure = plt.gca()
    figure.set_title(dataku_numerik.columns.values[i])
    jumlah_bins = np.size(dataku_numerik.iloc[:,i].unique())
    plt.hist(dataku_numerik.iloc[:,i], bins=jumlah_bins)

# Membuat Correlation Matrix
korelasi = dataku_numerik.drop(columns=['enrolled'], inplace=False).corrwith(dataku_numerik.enrolled)
korelasi.plot.bar(title='Korelasi Variabel Independen Terhadap Enrolled')
matriks_korelasi = dataku_numerik.drop(columns=['enrolled'], inplace=False).corr()
sns.heatmap(matriks_korelasi, cmap='Blues')

mask = np.zeros_like(matriks_korelasi, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Membuat Correlation Matrix Dengan Heatmap Custom
ax = plt.axes()
cmap_ku = sns.diverging_palette(200, 0, as_cmap=True)
sns.heatmap(matriks_korelasi, cmap=cmap_ku, mask=mask, linewidths=0.5, center=0, square=True)
ax = plt.suptitle('Correlation Matrix Custom')

# FEATURE ENGINEERING

# Proses Parsing
from dateutil import parser
dataku.first_open = [parser.parse(i) for i in dataku.first_open]
dataku.enrolled_date = [parser.parse(i) if isinstance(i, str) else i for i in dataku.enrolled_date]
dataku['selisih'] = (dataku.enrolled_date - dataku.first_open).astype('timedelta64[h]')

# Plotting dataku.selisih
plt.hist(dataku.selisih.dropna())
plt.suptitle('Selisih Waktu Antara Enrolled Dengan First Open')
plt.show()

# Mempersingkat range
plt.hist(dataku.selisih.dropna(), range=[0,200])
plt.suptitle('Selisih Waktu Antara Enrolled Dengan First Open')
plt.show()

# Memfilter Nilai Selisih > 48 Jam
dataku.loc[dataku.selisih>48, 'enrolled'] = 0

# Mengimpor Top Screens
top_screens = pd.read_csv('top_screens.csv')
top_screens = np.array(top_screens.loc[:,'top_screens'])

# Membuat Cadangan Data
dataku2 = dataku.copy()

# Membuat Kolom Untuk Setiap top_screens
for layar in top_screens:
    dataku2[layar] = dataku2.screen_list.str.contains(layar).astype(int)
    
for layar in top_screens:
    dataku2['screen_list'] = dataku2.screen_list.str.replace(layar+',','')
    
# Item Non top_screens di screen_list
dataku2['lainnya'] = dataku2.screen_list.str.count(',')

top_screens.sort()

# Proses Penggabungan Beberapa Screen Yang Sama (Funneling)
layar_loan = ['Loan',
              'Loan2',
              'Loan3',
              'Loan4']
dataku2['jumlah_loan'] = dataku2[layar_loan].sum(axis=1)
dataku2.drop(columns=layar_loan, inplace=True)

layar_saving = ['Saving1',
                'Saving2',
                'Saving2Amount',
                'Saving4',
                'Saving5',
                'Saving6',
                'Saving7',
                'Saving8',
                'Saving9',
                'Saving10']
dataku2['jumlah_saving'] = dataku2[layar_saving].sum(axis=1)
dataku2.drop(columns=layar_saving, inplace=True)

layar_credit = ['Credit1',
                'Credit2',
                'Credit3',
                'Credit3Container',
                'Credit3Dashboard']
dataku2['jumlah_kredit'] = dataku2[layar_credit].sum(axis=1)
dataku2.drop(columns=layar_credit, inplace=True)

layar_cc = ['CC1',
            'CC1Category',
            'CC3']
dataku2['jumlah_cc'] = dataku2[layar_cc].sum(axis=1)
dataku2.drop(columns=layar_cc, inplace=True)

# Mendefinisikan Var Dependen
var_enrolled = np.array(dataku2['enrolled'])

# Menghilangkan Beberapa Kolom Yang Redudan
dataku2.drop(columns = ['user', 'first_open', 'screen_list', 'enrolled', 'enrolled_date'], inplace=True)

dataku2.drop(columns = ['selisih'], inplace=True)

# Membagi Menjadi Training dan Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(dataku2), var_enrolled, test_size=0.2, random_state=111)

# Preprocessing Standardization (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Menghilangkan Variabel Kosong
X_train = np.delete(X_train, 27, 1)
X_test = np.delete(X_test, 27, 1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='liblinear', penalty = 'l1')
classifier.fit(X_train, y_train)

# Memprediksi Test Set
y_pred = classifier.predict(X_test)

# Confussion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Menggunakan Accuracy_Score
evaluasi = accuracy_score(y_test, y_pred)
print('Akurasi:{:.2f}'.format(evaluasi*100))

# Menggunakan Seaborn untuk CM
cm_label = pd.DataFrame(cm, columns=np.unique(y_test), index=np.unique(y_test))
cm_label.index.name = 'Aktual'
cm_label.columns.name = 'Prediksi'
sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')

# Validasi Dengan 10-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
print('Akurasi Regresi Logistik = {:.2f}% +/- {:.2f}%'.format(accuracies.mean()*100, accuracies.std()*100))

# Membagi Menjadi Training dan Test Set Untuk Cari UserID
var_enrolled = dataku2['enrolled']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataku2, var_enrolled, test_size=0.2, random_state=111)

train_id = X_train['user']
test_id = X_test['user']

# Menggabungkan Semua Data Prediksi
y_pred_series = pd.Series(y_test).rename('asli', inplace=True)
hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
hasil_akhir['prediksi'] = y_pred
hasil_akhir = hasil_akhir[['user', 'asli', 'prediksi']].reset_index(drop=True)