import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# MES FONTIONS
def vectoriser(data,num_classe):
 return np.eye(num_classe,dtype="int")[data]
#### PARTIE 1:Traitement des donnees:
## Telechargement des donnees
(X_train_data,Y_train_data),(X_test_data,Y_test_data)=tf.keras.datasets.mnist.load_data()
## Donnees d'apprentissage X
X_train=np.reshape(X_train_data,(X_train_data.shape[0],-1))
# norlamlisation la valeur des pixels sont dans [0,1]
X_train=X_train/255
print(f"x_train.shape{X_train.shape}")
# Pour valeur dans Y_train nous allons creer une liste de taille 10 ou les element sont dans {0,1}
Y_train=vectoriser(Y_train_data,10)
##  Donnees de test
X_test=np.reshape(X_test_data,(X_test_data.shape[0],-1))
X_test=X_test/255
# Pour valeur dans Y_test nous allons creer une liste de taille 10 ou les element sont dans {0,1}
Y_test=vectoriser(Y_test_data,10)

### PARTIE 2:reseau de neurones
# premieres couches entres de tailles(784=28*28)
modele=tf.keras.models.Sequential()
modele.add(tf.keras.layers.Dense(8,input_dim=784,activation='sigmoid'))
# DEUXIEME couche 8 neurones
modele.add(tf.keras.layers.Dense(units=8,activation='sigmoid'))
# COUCHE DE SORTIE 10 neurones
modele.add(tf.keras.layers.Dense(units=10,activation='softmax'))
# CHOIX DE LA METHODE DE DECENTE DE GRADIENT
modele.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# print(modele.summary())
# traitement
historic=modele.fit(X_train,Y_train,batch_size=32,epochs=37)

# RESULTATS
result=modele.evaluate(X_test,Y_test,verbose=0)
print(result)
# predict
y_predict=modele.predict(X_test)
print(f"np.argmax(y_predict[8]):{y_predict[8]}\n")
print(f"np.array(Y_test[8]): {np.array(Y_test[8])}\n")
print(f"np.argmax(y_predict[8]) :{np.argmax(y_predict[8])}\n")
print(f"np.argmax(np.array(Y_test[8])): {np.argmax(np.array(Y_test[8]))}\n")
# print(np.array(Y_test[8]))
fig,axe=plt.subplots(1,2)
# print(type(historic.history['loss']))
axe[0].plot(np.array(historic.history['loss']))
axe[1].imshow(X_test_data[8],cmap='Greys')
# plt.imshow(X_test_data[8],cmap='Greys')
plt.show()
# print(historic.history)



modele.save("chifre_classe.h5")

