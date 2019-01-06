# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:27:45 2018

@author: leu20
"""

"""
    比较两基金的α，看谁的大。
    比较两基金的β，看谁的波动大。
    比较夏普比率，看谁的大。
    组合内的个股分析
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)

#两基金2018年第三季度的成分股
df=pd.read_table('fundptfl.txt',encoding='utf-8')
df.head()
df=df.iloc[:,[1,2,6,7,8,9,10,11]]

len(np.unique(df['Symbol']))
np.unique(df['Symbol'])

#成分股第二季度日收益率
stkr=pd.read_table('stkreturn.txt',encoding='utf-8')
stkr.rename(columns={'Trddt':'date','Dretnd':'R'}, inplace=True) 
stkr.head()
#len(np.unique(stkr['Trddt']))


#市场组合(沪深300作为市场组合)第二季度日收益率
indexr=pd.read_csv('indexreturn.csv',index_col=0)
indexr=indexr.iloc[:,[1,4]]
indexr.rename(columns={'tradeDate':'date','return':'Rm'},inplace=True)

indexr.head()
len(indexr)

#获取无风险收益率
rf=pd.read_table('rf.txt',encoding='utf-8')
md=pd.DataFrame(np.unique(stkr['date']))
rf=rf.merge(md,left_on='Clsdt',right_on=0,how='inner',copy=False)
rf=rf.iloc[:,[1,2]]
rf.rename(columns={'Clsdt':'date','Nrrdaydt':'Rf'},inplace=True)

rf.head()
len(rf)


#所有数据整合到一个DataFrame
data=stkr.merge(indexr,on='date')
data=data.merge(rf,on='date')
data['Rme']=data['Rm']-data['Rf']
data['Re']=data['R']-data['Rf']
data=data.sort_values(by= ['Stkcd','date'],axis = 0,ascending = True)

data.head()
len(data)


#按照股票代码分组回归，求阿尔法和贝塔
grouped=data[['Rme','Re']].groupby(data['Stkcd'])
alpha=[]
beta=[]
for rme,re in grouped:
    est=sm.OLS(sm.add_constant(re['Rme']),re['Re']).fit()
    alpha.append([rme,est.params[0][0]])
    beta.append([rme,est.params[1][0]])
    print(est.params[1][0])
    print(rme)

alpha=pd.DataFrame(alpha)
beta=pd.DataFrame(beta)
alpha
beta
result=alpha.merge(beta,on=0)
result.rename(columns={0:'Stkcd','1_x':'alpha','1_y':'beta'},inplace=True)
result

#对result进行分析，查看哪支基金选的哪支股票
f1=df[df['MasterFundCode']==1044]['Symbol']
f1[0]
result['嘉实']='0'
for i in range(len(result)):
    if result.iloc[i,0] in list(f1):
        result.iloc[i,3]=1

f2=df[df['MasterFundCode']==110022]['Symbol']
result['易方达']='0'
for i in range(len(result)):
    if result.iloc[i,0] in list(f2):
        result.iloc[i,4]=1

name=df.iloc[:,[3,4]]
name

result=result.merge(name,left_on='Stkcd',right_on='Symbol')
len(result)
for i in range(20):
    result.iloc[i,1]=float('%.2f' %result.iloc[i,1])
for i in range(20):
    result.iloc[i,2]=float('%.2f' %result.iloc[i,2])

result


#计算两只基金的贝塔值
e=result.iloc[:,[5,2]]
e=e.drop_duplicates(keep='first')
dd=pd.merge(df,e,on='Symbol',how='inner')
dd=dd.iloc[:,[0,3,4,6,8]]
dd=dd.sort_values(by='MasterFundCode')
dd=dd.reset_index(drop=True)

group1=dd[['MarketValue','beta']].groupby(dd['MasterFundCode'])

for i,j in group1:
    sum1=sum(j['MarketValue'])
    value=sum(j['MarketValue']*j['beta']/sum1)
    value=float('%.2f' %value)
    print(i,value)


#计算处置效应和非处置效应
pf=pd.read_table('portfolio.txt',encoding='utf-8')
pf1=pf[pf['MasterFundCode']==1044]
pf1=pf1[pf1['ReportTypeID']<5]

pf1=pf1.iloc[:,[2,3,4,5,6,7,8]]
grouppf1=pf1.groupby(pf1['StockName'])
num=0
for i,j in grouppf1:
    a='2018-06-30' in list(j['EndDate'])
    b='2018-09-30' in list(j['EndDate'])
    c='2018-09-30' in list(j['EndDate'])
    if b:
        num+=1
        print(i)
        print("==================================")
        print(j.iloc[[-2,-1],:])
        print("**********************************")
        print(j.iloc[-1,5]-j.iloc[-2,5])
    
    
    
pf=pd.read_table('portfolio.txt',encoding='utf-8')
pf2=pf[pf['MasterFundCode']==110022]
pf2=pf2[pf2['ReportTypeID']<5]

pf2=pf2.iloc[:,[2,3,4,5,6,7,8]]
grouppf2=pf2.groupby(pf2['StockName'])
num2=0
for i,j in grouppf2:
    x='2018-06-30' in list(j['EndDate'])
    y='2018-09-30' in list(j['EndDate'])
    if y:
        num2+=1
        print(i)
        print("==================================")
        print(j)
        print("**********************************")
        try:
            print(j.iloc[-1,5]-j.iloc[-2,5])
        except:
            print(j.iloc[-1,5])


#获取成分股票的代码
st=list(np.unique(stkr['Stkcd']))
st=[str(i) for i in st]


stm=[]
for i in range(len(st)):
    if len(st[i])<6:
        n=6-len(st[i])
        stm.append('0'*n+st[i])
    else:
        stm.append(st[i])

#获取股票收盘价
clpr=pd.read_csv('stkcloseprice.csv',index_col=0)
clpr=clpr.iloc[:,[0,1,3,4,6]]
clpr['cha']=clpr['closePrice_y']-clpr['closePrice_x']
clpr


#将股票的涨跌趋势和基金里成分股信息合并
#==================================================================
#(第一只基金1044嘉实)
new1=pf1.merge(clpr[['ticker','cha']],left_on='Symbol',right_on='ticker')
new1.sort_values(by='EndDate')
new1=new1[(new1['EndDate']=='2018-09-30') | (new1['EndDate']=='2018-06-30')]
new1=new1.iloc[:,[1,3,4,5,6,8]]
new1=new1.reset_index(drop=True)
new1h=new1[new1['EndDate']=='2018-06-30']
new1t=new1[new1['EndDate']=='2018-09-30']
new1h
new1t
new1n=new1h.merge(new1t,on='Symbol',how='outer')
new1n

    #计算总亏损
gloss=0
for i in range(len(new1n)):
    if new1n.loc[i,'cha_x']<0:
        gloss=+new1n.loc[i,'Shares_x']*new1n.loc[i,'cha_x']

gloss
    #计算真实亏损
rloss=0
for j in range(len(new1n)):
    a=(new1n.loc[j,'cha_x']<0)
    b=(new1n.loc[j,'Shares_x']>new1n.loc[j,'Shares_y'])
    if (a & b):
        rloss=+(new1n.loc[j,'Shares_x']-new1n.loc[j,'Shares_y'])*new1n.loc[j,'cha_x']

rloss

    #计算总盈利
ggain=0
for i in range(len(new1n)):
    if new1n.loc[i,'cha_x']>0:
        ggain=+new1n.loc[i,'Shares_x']*new1n.loc[i,'cha_x']

ggain



    #计算真实盈利
rgain=(new1n.loc[3,'Shares_x'])*new1n.loc[3,'cha_x']
for j in range(len(new1n)):
    a=(new1n.loc[j,'cha_x']>0)
    b=(new1n.loc[j,'Shares_x']>new1n.loc[j,'Shares_y'])
    if (a & b):
        rgain=+(new1n.loc[j,'Shares_x']-new1n.loc[j,'Shares_y'])*new1n.loc[j,'cha_x']

rgain
    #计算处置效应
d1=rgain/ggain-rloss/gloss
d1

#(第二只基金110022易方达)
new2=pf2.merge(clpr[['ticker','cha']],left_on='Symbol',right_on='ticker')
new2.sort_values(by='EndDate')
new2=new2[(new2['EndDate']=='2018-09-30') | (new2['EndDate']=='2018-06-30')]
new2=new2.iloc[:,[1,3,4,5,6,8]]
new2=new2.reset_index(drop=True)
new2h=new2[new2['EndDate']=='2018-06-30']
new2t=new2[new2['EndDate']=='2018-09-30']
new2h
new2t
new2n=new2h.merge(new2t,on='Symbol',how='outer')
new2n

    #计算总亏损
gloss2=0
for i in range(len(new2n)):
    if new2n.loc[i,'cha_x']<0:
        gloss2=+new2n.loc[i,'Shares_x']*new2n.loc[i,'cha_x']

gloss2
    #计算真实亏损
rloss2=0
for j in range(len(new2n)):
    a=(new2n.loc[j,'cha_x']<0)
    b=(new2n.loc[j,'Shares_x']>new2n.loc[j,'Shares_y'])
    if (a & b):
        rloss2=+(new2n.loc[j,'Shares_x']-new2n.loc[j,'Shares_y'])*new2n.loc[j,'cha_x']

rloss2

    #计算总盈利
ggain2=0
for i in range(len(new2n)):
    if new2n.loc[i,'cha_x']>0:
        ggain2=+new2n.loc[i,'Shares_x']*new2n.loc[i,'cha_x']

ggain2



    #计算真实盈利
rgain2=0
for j in range(len(new2n)):
    a=(new2n.loc[j,'cha_x']>0)
    b=(new2n.loc[j,'Shares_x']>new2n.loc[j,'Shares_y'])
    if (a & b):
        rgain2=+(new2n.loc[j,'Shares_x']-new2n.loc[j,'Shares_y'])*new2n.loc[j,'cha_x']

rgain2

    #计算处置效应
d2=rgain2/ggain2-rloss2/gloss2
d2

















