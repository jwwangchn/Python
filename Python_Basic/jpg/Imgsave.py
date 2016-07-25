#coding:utf8
#爬虫
#获取 https://www.zhihu.com/question/32759798中的图片
import re
import urllib

def getHtml(url):
    page=urllib.urlopen(url)
    html=page.read()
    return html

def getImage(html):
    reg=r'img src="(https.*?\.png)"'
    imgre=re.compile(reg)
    imglist = re.findall(imgre,html)
    x=0
    for imgurl in imglist:
        urllib.urlretrieve(imgurl,'%s.png' % x)
        x+=1
        print '第 %s 张图片' % x

html=getHtml('https://www.zhihu.com/question/32759798')
getImage(html)