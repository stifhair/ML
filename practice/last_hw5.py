import json
data = '''{
    "name": "Chuck",
    "phone" : {
        "type" : "intl",
        "number" : "12 1232 12312 123"
        },
        "email":{
            "hide":"yes"
            }
    }
'''

info = json.loads(data)
print ('Name:',info["name"])
print ('Hide:',info["email"]["hide"])