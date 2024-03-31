import pandas as pd
from googletrans import Translator
import re
import string

def translate_text(text, dest='en'):
    if text is None or text.strip() == "":
        return ""  # hoặc bạn có thể trả về text ban đầu hoặc thực hiện một hành động khác tuỳ thuộc vào yêu cầu của bạn
    translator = Translator()
    translation = translator.translate(text, dest=dest)
    return translation.text

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Label'] = data['Label'].map({'normal': 0, 'defaced': 1})
    data.head()
    X = data['Content'].apply(wordopt)
    y = data['Label']
    print("Số lượng dữ liệu:", len(X))
    return X, y
