import requests
from bs4 import BeautifulSoup
import time
import csv
 
need_reviews_cnt = 500
reviews = []
review_data=[]

#page를 1부터 1씩 증가하며 URL을 다음 페이지로 바꿈
#page 1 2 3 4 버튼을 클릭하면 url이 변하는 형식이 아니라 1페이지 리뷰 20개가 반복됨. 
#제목과 내용별로 줄 나눠서 표현하려했는데 실패함.
#.csv파일을 열려고 하니 파일에 문제가 생겨서 open불가. vscode에서만 확인가능 
for page in range(1,5):
    url = f'https://search.shopping.naver.com/catalog/30819376142?NaPm=ct%3Dl1k0fb0w%7Cci%3D54ea254aa34b5942102782dd0b7bf430722a9887%7Ctr%3Dslcc%7Csn%3D95694%7Chk%3Dce78eab6e96197231466bf0aeada943a2a2a8b0f'
    #get : request로 url의  html문서의 내용 요청
    html = requests.get(url)
    #html을 받아온 문서를 .content로 지정 후 soup객체로 변환
    soup = BeautifulSoup(html.content,'html.parser')
    #find_all : 지정한 태그의 내용을 모두 찾아 리스트로 반환
    reviews = soup.find_all("div",{"class":"reviewItems_review_text__2Bwpa"})
    
    #한 페이지의 리뷰 리스트의 리뷰를 하나씩 보면서 데이터 추출
    for review in reviews:
        sentence = review.find("em",{"class":"reviewItems_title__39Z8H"}).get("onclick")#.split("', '")[2]
        #Line22에서 NoneType object has no attribute 'split' 오류발생... 왜...?
        if sentence != "":
            reviewTitle = review.find("em",{"class":"reviewItems_title__39Z8H"}).get_text()
            Texts = review.find("p",{"class":"reviewItems_text__XIsTc"}).get_text()
            review_data.append([reviewTitle,sentence,Texts])
            need_reviews_cnt-= 1
    #현재까지 수집된 리뷰가 목표 수집 리뷰보다 많아진 경우 크롤링 중지        
    if need_reviews_cnt < 0:                                         
        break
    #다음 페이지를 조회하기 전 0.5초 시간 차를 두기
    time.sleep(0.5)
     
columns_name = ["review"]
with open ( "samples4.csv", "w", newline ="",encoding = 'utf8' ) as f:
    write = csv.writer(f)
    write.writerow(columns_name)
    write.writerows(review_data)
 

