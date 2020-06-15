import numpy
import pandas as pd
from matplotlib import pyplot as plt

res_table = pd.read_excel('./sample.xlsx',
                            sheet_name ='output'
                           ,header = 0)

row_table = pd.read_excel('./sample.xlsx',
                           sheet_name ='input',
                           header = 0)

std_res = list(res_table.columns)
res_name =list(res_table.iloc[0])
del res_name[0]


# 일단 수익률 테이블을 만들어서 넣는다.
index        = list(row_table.columns)
index_header = list(row_table.columns)
index_header.append("portfolio_yield")
del index[0]
yield_lst     = []
yield_lst_set = []
yield_table  = pd.DataFrame(columns=index_header)
port_yield   = 0.0

#행의 개수만큼 돌린다. 수익률을 구한다.
for i in range( 1, row_table.shape[0]-1 ):
    yield_lst = []
    port_yield = 0.0
    yield_lst.append(row_table.iloc[i + 1]['date'])
    for index_name in index:
        imsi= row_table.iloc[i + 1][index_name] / row_table.iloc[i][index_name] - 1
        yield_lst.append(imsi)
        if index_name != 'MXWD INDEX' and index_name != 'USDKRW Curncy' :
            port_yield = port_yield + (imsi) * row_table.iloc[0][index_name]
        else:
            pass
    yield_lst.append(port_yield)
    yield_lst_set =[]
    yield_lst_set.append(tuple(yield_lst))
    df2 = pd.DataFrame.from_records(yield_lst_set,columns=index_header)
    yield_table=yield_table.append(df2)

yield_table = yield_table.reset_index(drop=True)

for i in range(0, len(std_res)):
    str_expr1 = "date=={}".format(std_res[i])
    str_expr2 = "date=={}".format(std_res[i+1])
    df1 = yield_table.query(str_expr1)
    df2 = yield_table.query(str_expr2)
    for index_name in index:
        imsi = df1.iloc[0][index_name]/df2.iloc[0][index_name] - 1







print (row_table.head())
print (row_table.tail())


