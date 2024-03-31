from sklearn.model_selection import train_test_split
from data.load_data import load_data
from data.preprocess_data import preprocess_data
from models.build_model import build_model
from models.train_model import train_model
from models.evaluate import evaluate_model

if __name__ == "__main__":
    file_path = 'data/data.csv'
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_padded, X_test_padded, word_index = preprocess_data(X_train, X_test)
    vocab_size = len(word_index) + 1
    embedding_dim = 128
    # max_length = 128
    model = build_model(vocab_size, embedding_dim)
    trained_model= train_model(model, X_train_padded, y_train, X_test_padded, y_test)
    evaluate_model(trained_model, X_test, y_test)
