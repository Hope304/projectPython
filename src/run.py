import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.getNewContent import get_webpage_content
from data.load_data import translate_text
from data.load_data import wordopt
from data.preprocess_data import  preprocess_data
def predict_from_link(link, model):
    # Đọc dữ liệu từ đường link
    X_data = get_webpage_content(link)
    if X_data:
        X_data = wordopt(X_data)
        X_data = translate_text(X_data)
        print(X_data)
        # Dự đoán kết quả
        predictions = model.predict(X_data)
        return predictions
    return X_data

if __name__ == "__main__":
# Lấy đường dẫn của thư mục chứa mã hiện tại (src)
    src_directory = os.path.dirname(os.path.abspath(__file__))
    project_directory = os.path.dirname(src_directory)
    file_path = os.path.join(project_directory, 'model.keras')
    model = load_model(file_path)
    model.summary()
    link = input("Nhập đường link: ")
    predictions = predict_from_link(link)
    print(predictions)
