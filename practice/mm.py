import urllib.request
import pandas as pd
from pandas.io.json import json_normalize
import json
import xmltodict


def getData(parameter):
    request  = urllib.request.Request(parameter)
    response = urllib.request.urlopen(request)
    rescode  = response.getcode()

    response_body = response.read().decode('utf-8')
    result = response_body
    return result

url = """http://openapi.gbis.go.kr/ws/rest/busarrivalservice?serviceKey=1234567890&stationId=200000177&routeId=200000037&staOrder=19&type=json
"""

xmlString =getData(url)
jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)
print(jsonString)

json_object = json.loads(jsonString)




print(json_object)
