import pandas as pd  # 匯入 pandas 模組，用於資料處理和匯出 Excel
from bs4 import BeautifulSoup  # 匯入 BeautifulSoup 模組，用於網頁解析
import requests  # 匯入 requests 模組，用於發送網路請求
import re  # 匯入 re 模組，用於正則表達式操作
import time  # 匯入 time 模組，用於延遲等待

keyword = '貓砂'  # 輸入搜尋關鍵字
pages = 100  # 總頁數為100
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
# 使用者代理設定，模擬瀏覽器請求
base_url = 'https://m.momoshop.com.tw/search.momo?searchKeyword={}&cpCode=&couponSeq=&searchType=1&cateLevel=-1&curPage={}&cateCode=&cateName=&maxPage=77&minPage=1&_advCp=N&_advFirst=N&_advFreeze=N&_advSuperstore=N&_advTvShop=N&_advTomorrow=N&_advNAM=N&_advStock=N&_advPrefere=N&_advThreeHours=N&_advVideo=N&_advCycle=N&_advCod=N&_advSuperstorePay=N&_advPriceS=&_advPriceE=&_brandNameList=&_brandNoList=&brandSeriesStr=&isBrandSeriesPage=0&ent=b&_imgSH=itemizedType&specialGoodsType=&_isFuzzy=0&_spAttL=&_mAttL=&_sAttL=&_noAttL=&topMAttL=&topSAttL=&topNoAttL=&hotKeyType=0&hashTagCode=&hashTagName=&serviceCode=MT01'
# 搜尋基本URL，包含動態搜尋參數

urls = set()  # 使用 set 來存儲連結，自動去重

for page in range(1, pages + 1):  # 遍歷總頁數，進行網頁爬取
    url = base_url.format(keyword, page)  # 構建具體的搜尋頁面URL
    resp = requests.get(url, headers=headers)  # 發送網路請求
    if resp.status_code == 200:  # 若請求成功
        soup = BeautifulSoup(resp.text, features="html.parser")  # 使用 BeautifulSoup 解析HTML內容
        for item in soup.select('li.goodsItemLi > a'):  # 選擇所有商品連結元素
            if keyword in item['href']:  # 添加過濾條件，只添加包含「貓砂」關鍵字的連結
                urls.add('https://m.momoshop.com.tw' + item['href'])  # 添加商品連結到 set 中
        time.sleep(1)  # 延遲1秒，避免過快訪問

urls = list(urls)  # 將 set 轉換為 list，便於後續操作

# 輸出結果
df = []  # 建立空的列表，用於存放 DataFrame

for i, url in enumerate(urls):  # 遍歷所有商品連結，進行商品詳細資訊爬取
    columns = []  # 存放欄位名稱
    values = []  # 存放每個商品的資訊
    
    resp = requests.get(url, headers=headers)  # 發送商品詳細資訊頁面的網路請求
    soup = BeautifulSoup(resp.text, features="html.parser")  # 使用 BeautifulSoup 解析HTML內容
    # 標題
    title = soup.find('meta',{'property':'og:title'})  # 找到標籤屬性為 og:title 的元素
    title = title['content'] if title else '未知'  # 如果找到標題，取得其內容，否則設定為未知
    # 品牌
    brand = soup.find('meta',{'property':'product:brand'})  # 找到標籤屬性為 product:brand 的元素
    brand = brand['content'] if brand else '未知'  # 如果找到品牌，取得其內容，否則設定為未知
    # 原價
    try:
        price = re.sub(r'\r\n| ','',soup.find('del').text)  # 清除原價文字中的換行符和空格
    except:
        price = ''  # 若無原價資訊，設定為空字串
    # 特價
    amount = soup.find('meta',{'property':'product:price:amount'})  # 找到標籤屬性為 product:price:amount 的元素
    amount = amount['content'] if amount else '未知'  # 如果找到特價，取得其內容，否則設定為未知
    # 類型
    cate = ''.join([i.text for i in soup.findAll('article',{'class':'pathArea'})])  # 找到所有類型元素，並將文本連接成字串
    cate = re.sub('\n|\xa0',' ',cate)  # 使用正則表達式替換文本中的換行符和特殊符號
    # 描述
    desc = soup.find('div',{'class':'Area101'})  # 找到描述區域元素
    desc = re.sub('\r|\n| ', '', desc.text) if desc else '無描述'  # 如果找到描述，清除文本中的換行符和空格，否則設定為無描述
    
    print('==================  {}  =================='.format(i))    
    print("標題:",title)
    print("品牌:",brand)
    print("原價:",price)
    print("特價:",amount)
    print("類別:",cate)
    print("描述文字:",desc)  
    columns += ['title', 'brand', 'price', 'amount', 'cate', 'desc']  # 更新欄位名稱列表
    values += [title, brand, price, amount, cate, desc]  # 更新每個商品的資訊列表

    # 將 values 轉換為 DataFrame
    ndf = pd.DataFrame(data=[values], columns=columns)  # 創建新的 DataFrame
    df.append(ndf)  # 將 DataFrame 加入到列表中
    time.sleep(2)  # 增加延遲時間為2秒      
# 使用 pd.concat 將列表中的 DataFrame 合併為一個 DataFrame
df = pd.concat(df, ignore_index=True)  # 合併所有 DataFrame
df.to_excel('momoshop.xlsx', index=False)  # 將 DataFrame 匯出成 Excel 檔案，不包含索引列
