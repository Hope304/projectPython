from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_prob = model.predict(X_test) 
    y_pred = (y_pred_prob > threshold).astype(int) 
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
