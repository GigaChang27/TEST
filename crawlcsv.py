import pandas as pd
from bs4 import BeautifulSoup
import requests
import time

base_url = 'https://www.thb.gov.tw/NewsConstructionRoad.aspx?n=353&page={}&PageSize=10'


def fetch_construction_data(base_url, total_pages):
    data = []  # 存放施工地點與日期

    for page in range(1, total_pages + 1):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        construction_items = soup.find_all('tr')  # 抓取全部tr相關的標題

        for item in construction_items:
            location_element = item.find('td', class_='CCMS_jGridView_td_Class_0')  # 獲取地點元素
            date_element = item.find('td', class_='CCMS_jGridView_td_Class_1')  # 獲取日期元素
            if location_element and date_element:
                location = location_element.text.strip()  # 提取地點
                date = date_element.text.strip()  # 提取發布日期
                data.append({'施工地點': location, '發布日期': date})  # 添加到數據列表
                print('施工地點:', location)
                print('發布日期:', date)
                print()  # 空行分隔每筆資料的打印

        time.sleep(1)  # 1秒休息，避免請求過快

    return data

# 設定要抓取的總頁數
total_pages_to_fetch = 28 

# 獲取数据
data = fetch_construction_data(base_url, total_pages_to_fetch)

# 將數據轉成df
df = pd.DataFrame(data)

# 匯出Excel
df.to_excel('construction_data.xlsx', index=False)
