import requests
from bs4 import BeautifulSoup

def get_webpage_content(link):
    url = link
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    except requests.exceptions.RequestException as e:
        print('Không thể kết nối đến trang web:', e)
        return None
    except Exception as e:
        print('Đã xảy ra lỗi:', e)
        return None
