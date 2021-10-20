from selenium import webdriver
import time
url = "https://music.yandex.ru/"
driver = webdriver.Firefox(executable_path="C:\\PythonProject\\Project_one\\webdriver\\geckodriver.exe")

try:
    driver.get(url)
    time.sleep(5)
except Exception as ex:
    print(ex)
finally:
    # driver.close()
    # driver.quit()
    pass