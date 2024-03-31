from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=32, model_path="model.keras"):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Lưu mô hình tốt nhất dựa trên val_accuracy
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # Huấn luyện mô hình
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[checkpoint])
    model.summary()
    return model
