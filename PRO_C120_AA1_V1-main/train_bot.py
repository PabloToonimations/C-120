#biblioteca para entrenamiento del modelo
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import dense, activation, dropout
from tensorflow.keras.optimizer import adam

def train_bot_model (train_x,train_y):
	model=sequential()
	model.add(dense(128,input_shape=(len(train_x[0]),),activation='relu'))
	model.add(dropout(0.5))
	model.add(dense(64,activation='relu'))
	model.add(dropout(0.5))
	model.add(dense(len(train_y[0]),activation='softmax'))

	#compilar el modelo
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
