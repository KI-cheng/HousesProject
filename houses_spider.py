from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import seleniumwire.undetected_chromedriver as uc
from fake_useragent import UserAgent
from lxml import html
import csv
import hashlib
import pandas as pd

etree = html.etree


def get_Page_Data(url):
    # ------------ 规避检测 ------------
    # 实例化对象
    option = uc.ChromeOptions()
    option.add_experimental_option('excludeSwitches', ['enable-automation'])  # 开启实验性功能
    # 去除特征值
    option.add_argument("--disable-blink-features=AutomationControlled")
    # 隐藏
    option.add_argument('--headless')
    # 实例化谷歌
    driver = webdriver.Chrome(options=option)
    # 获取页面源码
    driver.get(url)

    try:
        flag = WebDriverWait(driver, 10)
    except:
        return 'error'
    page_text = driver.page_source
    # 解析
    return page_text


def id_md5(date, region, address, position):
    combine = f"{date}{region}{address}{position}"
    obj = hashlib.md5()
    obj.update(combine.encode(encoding='utf-8'))
    md5_hash = obj.hexdigest()
    return md5_hash


def get_message_from_page(page_text):
    # 测试后发现：租房和卖房信息共区，按照间隔区分即可
    # 格式：date(0), region(1), --, --, address(4), --, --,
    #      tag(待处理)(9),-->location/floor/room
    #      price(10),
    #      area_real(11), area_built(11) (ft.)需要清洗
    #      price_per_foot_real(12), price_per_foot_built(12) (HKD)需要清洗
    #      source (18)
    html = etree.HTML(page_text)
    all_houses = []
    tbody_element = html.find(".//tbody")
    if tbody_element is not None:
        tr_elements = tbody_element.findall(".//tr")
        for tr in tr_elements:
            elements = tr.findall(".//")  # 这里包含了div和tr下的td，格式固定
            house = []  # 暂存一组信息
            house.append(id_md5(elements[0].text, elements[1].text, elements[4].text, elements[9].text))
            for i in [0, 1, 4, 9, 10, 11, 12, 18]:
                if i == 9:
                    tags = elements[i].text.split(" ")
                    # 观察得知无论无何中间是描述中高低层的，左右两边分别是描述位置（座）和房间（室）
                    location = 'NA'
                    floor = 'NA'
                    room = 'NA'
                    flag = 0
                    for tag in tags:
                        if '層' in tag:
                            flag = 1
                            if tag == '低層':
                                floor = 'low'
                            elif tag == '中層':
                                floor = 'middle'
                            elif tag == '高层':
                                floor = 'high'
                        elif '室' in tag:
                            room = tag.replace('室', '')
                            flag = 1
                    # 处理剩余情况
                    if flag == 0:  # 都没有相符信息，剩下的是location
                        location = ''.join(tags)
                    else:  # 有相符信息，剔除含有‘室’和‘层’的，拼接剩下的location
                        filtered_list = [item for item in tags if not any(char in item for char in ['室', '層'])]
                        location = ''.join(filtered_list)
                    house.append(location)
                    house.append(floor)
                    house.append(room)
                elif i == 11:
                    areas = elements[i].text.replace('呎', '').replace('--', 'NA').split(" / ")
                    house.append(areas[0])
                    house.append(areas[1])
                elif i == 12:
                    prices = elements[i].text.replace('$', '').replace('--', 'NA').split(" / ")
                    house.append(prices[0])
                    house.append(prices[1])
                elif i == 18:
                    house.append(
                        elements[i].text.replace(' ', '').replace('*', '').replace('\n', '').strip('"').strip(' '))

                else:
                    house.append(elements[i].text)
            print(house)
            all_houses.append(house)
            print("------------------------------------------------------------------------------"
                  "------------------------------------------------------------------------------")
    return all_houses


def save_data(all_houses):
    with open('rent_houses.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_houses)
    f.close()


def start_csv():
    column_names = "id,date,region,address,location,floor,room,sold_price(HKD)" \
                   ",actual_area(ft.),built_area(ft.),actual_price(HKD/ft.),built_price(HKD/ft.),source"
    with open('rent_houses.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(column_names.split(','))
    f.close()


def start_spider(x=1, y=1001):
    for i in range(x, y):
        url = f"https://www.house730.com/deal/g{i}t2/?type=rent"
        page_text = get_Page_Data(url=url)
        if page_text == 'error':
            continue
        try:
            all_houses = get_message_from_page(page_text)
        except:
            continue
        save_data(all_houses)
        print(f"已爬取第{i}页")


def washing():
    df = pd.read_csv('rent_houses.csv')
    results = df.drop_duplicates(subset='id', keep='first')
    df_filtered = results[~results['sold_price(HKD)'].astype(str).str.contains('萬')]
    df_cleaned = df_filtered.dropna()
    df_cleaned.to_csv('rent_houses_washed2.csv', index=False)


if __name__ == '__main__':
    # start_csv()
    # start_spider()
    washing()
