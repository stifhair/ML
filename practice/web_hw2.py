import urllib.request, urllib.parse, urllib.error
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

address = input("Enter location: ")

param = dict()
param["address"] = address
param['key'] = 42

print(urllib.parse.urlencode(param))

url = "http://py4e-data.dr-chuck.net/json?" + urllib.parse.urlencode(param)
print(url)

uh = urllib.request.urlopen ( url, context=ctx)
data= uh.read().decode()

js = json.loads(data)
lat = js['results'][0]['geometry']['location']['lat']
lng = js['results'][0]['geometry']['location']['lng']
print(js['results'][0]['geometry']['location'])
print(address)
print("lat",lat,"lng",lng)

try:
    js = json.loads(data)
except:
    js = None

if not js or 'status' not in js or js['status'] != 'OK':
    print('==== Failure To Retrieve ====')
    print(data)

counter = -1
info = js["results"][0]["address_components"]
for item in info:
    counter += 1
    if js["results"][0]["address_components"][counter]["types"] == ['country', 'political']:
        print(js["results"][0]["address_components"][counter]["short_name"])
    else:
        continue

