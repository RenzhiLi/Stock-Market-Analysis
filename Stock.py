import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
from pandas_datareader.data import DataReader
from datetime import datetime
com_list=['AAPL','GOOG','MSFT','AMZN']
end=datetime.now()
start=datetime(end.year-2,end.month,end.day)
'''
for stock in com_list:
	globals()[stock]=DataReader(stock,'yahoo',start,end)
print(AAPL.tail())

AAPL['Adj Close'].plot(legend=True,figsize=(15,6))
plt.show()
AAPL['Volume'].plot(legend=True,figsize=(15,6))
plt.show()
ma_day=[5,20,60]
for ma in ma_day:
	column_name=f"MA for {str(ma)} days"
	AAPL[column_name]=AAPL['Adj Close'].rolling(ma).mean()
AAPL[['Adj Close','MA for 5 days','MA for 20 days','MA for 60 days']].plot(subplots=False,legend=True,figsize=(15,6))
plt.show()

AAPL['Daily Return']=AAPL['Adj Close'].pct_change()
#AAPL['Daily Return'].plot(figsize=(15,6),legend=True)
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='green')
plt.show()
'''
closing_df=DataReader(com_list,'yahoo',start,end)['Adj Close']
tech_rtrn=closing_df.pct_change()
rets=tech_rtrn.dropna()
'''
closing_df.plot(figsize=(15,6))
plt.show()
#sns.jointplot('AAPL','AMZN',tech_rtrn,kind='scatter',s=3)
rtrnfig=sns.PairGrid(closing_df)
rtrnfig.map_diag(sns.distplot,bins=40,color='green')
rtrnfig.map_upper(plt.scatter,s=2)
rtrnfig.map_lower(sns.kdeplot,cmap='coolwarm')
plt.show()

corr=closing_df.dropna().corr()
sns.heatmap(corr,annot=True,center=0.5,cmap='coolwarm')
plt.show()

area=np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected returns')
plt.ylabel('Risk')
plt.xlim(rets.mean().min()*0.9,rets.mean().max()*1.1)
plt.ylim(rets.std().min()*0.9,rets.std().max()*1.1)
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
	plt.annotate(label,xy=(x,y))
plt.show()
'''
#print(rets['MSFT'].quantile(0.01))
def stock_monte_carlo(start_price,days,mu,sigma):
    #partially retrieved from internet
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    # Define a price array
    dt=1/days
    price = np.zeros(days)
    price[0] = start_price
    # Shock and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)   
    # Run price array for number of days
    for x in range(1,days):       
        # Calculate Shock
        shock[x] = np.random.normal(loc=0, scale=sigma)
        # Calculate Drift
        drift[x] = mu
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))        
    return price
start_price=107.12
days=503
mu=rets.mean()['AAPL']
sigma=rets.std()['AAPL']

for run in range(40):
	plt.plot(stock_monte_carlo(start_price,days,mu,sigma),linewidth=0.8)
plt.plot(np.array(closing_df['AAPL']),label='Actual',linewidth=4,color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
'''
actual_final=204.47
runs=1000
simulations=np.zeros(runs)
np.set_printoptions(threshold=5)
for run in range(runs):
	simulations[run]=stock_monte_carlo(start_price,days,mu,sigma)[days-1]
q=np.percentile(simulations,1)
sim_mean=simulations.mean()
plt.hist(simulations,bins=200)
plt.axvline(x=start_price,linewidth=3,color='brown')
plt.figtext(0.6,0.8,s=f'Start price: {start_price}',color='brown')
plt.axvline(x=q,linewidth=3,color='r')
plt.figtext(0.6,0.5,s=f'q(99%): {q:{5}.{5}}',color='r')
plt.figtext(0.6,0.45,s=f'Gain (q(99%)): {q-start_price:{4}.{4}}',color='r')
plt.figtext(0.6,0.4,s=f'Gain (q(99%)): {(q-start_price)*100/start_price:{4}.{4}}%',color='r')
plt.axvline(x=actual_final,linewidth=3,color='green')
plt.figtext(0.6,0.7,s=f'Actual final price: {actual_final}',color='green')
plt.axvline(x=sim_mean,linewidth=3,color='purple')
plt.figtext(0.6,0.6,s=f'Mean final price: {sim_mean:{5}.{5}}',color='purple')
plt.title("Price Distribution with 10,000 Random Runs")
plt.show()
'''