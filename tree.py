# Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Cargar los datos que se van a utilizar
df = pd.read_csv('nasa.csv')

# Evaluar las variables que presentan mayor correlación con la variable que busco predecir.
# Esto se realiza en diferentes matrices de correlación para tener una mejor visualización
corr_matrix = df[['Hazardous', 'Absolute Magnitude', 'Est Dia in KM(min)',
       'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
       'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
       'Est Dia in Feet(min)', 'Est Dia in Feet(max)']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot = True, fmt='.4f', linewidths=0.5)

corr_matrix = df[['Hazardous', 'Relative Velocity km per sec',
       'Relative Velocity km per hr', 'Miles per hour',
       'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)',
       'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body', 'Orbit Uncertainity',
       'Minimum Orbit Intersection']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot = True, fmt='.4f', linewidths=0.5)

corr_matrix = df[['Hazardous', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot = True, fmt='.4f', linewidths=0.5)

corr_matrix = df[['Hazardous', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion', 'Equinox']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot = True, fmt='.4f', linewidths=0.5)
plt.show()

# Extraer los datos seleccionados gracias a las matrices anteriores en dos df, mis variables independientes y la dependiente
datafile = df[['Absolute Magnitude','Est Dia in KM(min)','Est Dia in KM(max)',
               'Relative Velocity km per hr','Miss Dist.(Astronomical)','Orbit Uncertainity',
               'Minimum Orbit Intersection','Epoch Osculation','Eccentricity','Semi Major Axis',
               'Asc Node Longitude','Orbital Period','Perihelion Distance','Aphelion Dist',
               'Perihelion Time', 'Mean Anomaly','Mean Motion']]
labelsfile = df[['Hazardous']]

# Reemplazar los valores booleanos por strings de la variable dependiente
# Corre - representa que el asteroide es peligroso
# Todo bien -  el asteroide no representa un peligro
labelsfile = labelsfile.replace([True],'Corre')
labelsfile = labelsfile.replace([False],'Todo bien')

# Trsnaformas mis etiquetas en números, en este caso 0 y 1
# 0 - Corre, 1 - Todo bien
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(labelsfile)

# Separar los datos en datos de entrenamiento y en datos de prueba 
x_train = datafile[:4219]
y_train = pd.DataFrame(true_labels[:4219], columns=['Hazardous'])
x_test = datafile[4219:]
y_test = pd.DataFrame(true_labels[4219:], columns=['Hazardous'])

# Creación del árbol
tree_clf = DecisionTreeClassifier(max_depth = 6)
tree_clf.fit(x_train, y_train)

# Visualización del árbol
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(tree_clf, 
                   feature_names=datafile.columns.values,  
                   class_names=label_encoder.classes_,
                   filled=True)
fig.savefig("decistion_tree.png")
plt.show()

# Predicciones
pred =  tree_clf.predict(x_test)
testing = pd.DataFrame()
testing["Hyp y"] = pred
testing["Real y"] = y_test['Hazardous'].values
print(testing)

# Gráfica de los valores hipotéticos de x contra los reales
x = np.array(range(0,len(x_test-1)))
plt.scatter(x, testing["Hyp y"].values)
plt.scatter(x, testing["Real y"].values)
plt.show()

# Accuracy de mi train y de mi test para descartar overfitting
print("Train accuracy", tree_clf.score(x_train, y_train))
print("Test accuracy", accuracy_score(testing["Real y"].values, testing["Hyp y"].values))