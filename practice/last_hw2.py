import urllib.request , urllib.parse, urllib.error
import re
import ssl
import bs4 import BeautifulSoup

ctx = ssl.create.default.context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url  = "http://prl.korea.ac.kr/~pronto/home/index.php"
html = urllib.request.urlopen(url,context=ctx).read()

soup = BeautifulSoup ( html,"html.parser")
tags = soup('a')

for tag in tags:
    print(tag.get('href',None))

#links = re.findall(b'href="(http[s]?://.*?)"',html)
#for link in links:
#    print(link.decode())

