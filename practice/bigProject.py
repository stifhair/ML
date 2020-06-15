import numpy
import pandas as pd

row_table = pd.read_excel('./sample.xlsx',
                     sheet_name ='input',
                     header = 0)

# 일단 수익률 테이블을 만들어서 넣는다.
index        = list(row_table.columns)
index_header = list(row_table.columns)
index_header.append("portfolio_yield")
del index[0]
yield_lst    = []
yield_table  = pd.DataFrame(columns=index_header)


#행의 개수만큼 돌린다. 수익률을 구한다.
for i in range( 1, row_table.shape[0] +1 ):
    yield_lst = []
    yield_lst.append(row_table.iloc[i + 1]['date'])
    for index_name in index:
        yield_lst.append(row_table.iloc[i+1][index_name] / row_table.iloc[i][index_name] - 1)
    yield_table.append(yield_lst)

print (row_table.head())
print (row_table.tail())    ########
