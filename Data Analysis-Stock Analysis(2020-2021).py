#!/usr/bin/env python
# coding: utf-8

# <h1>Data Analysis Project:Stock Market Analysis</h1>
# <h2>ANANTHA SAI AVADHANAM</h2>
# <h2>23/05/2021</h2>

# <p>In this Notebook,We will analyse and try to get some insights between the trends of stocks related to some organizations.</p>
# <p>We will use pandas_datareader from pandas to get Stock market data from Yahoo Finance.</p>
# <br>
# <p>We will try to find answers for the following questions.</p>
# <ul><p>What is the behaviour of stock's price over time?<br>
# What is the daily return average of a stock?<br>
# What is the moving average of various stocks?<br>
# What is the correlation between Returns of different stocks?<br>
# What is the Value at risk by investing in a particular stock?<br>
# What is the predicted future stock behaviour for a particular stock?</p></ul>

# <h4>We will analyze the stock market data for the following companies from May,2020 to May,2021.</h4>
# <ul><h4>Tesla<br>Rolls-Royce<br>Walmart<br>Xiomi<br>Alibaba<br>Netflix<br>Google</h4></ul>

# In[94]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from datetime import datetime
sns.set_style('whitegrid')


# In[16]:


from __future__ import division
from pandas_datareader import data


# <p>We will use tickers to get the required data.</p>
# <p>A ticker symbol is an abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market.</p>

# In[17]:


#TSLA=Tesla
#RYCEY=Rolls-Royce
#WMT=Walmart
#XIACF=Xiomi
#BABA=Alibaba
#NFLX=Netflix
#GOOG=Google
companies=['TSLA','RYCEY','WMT','XIACF','BABA','NFLX','GOOG']


# In[18]:


end_time=datetime.now()
start_time=datetime(end_time.year-1,end_time.month,end_time.day)
print("From",start_time)
print("To",end_time)


# In[21]:


for company in companies:
    globals()[company]=data.DataReader(company,'yahoo',start_time,end_time)


# <h4> Let's take a look at our data.</h4>

# In[28]:


TSLA.tail(2)


# In[29]:


RYCEY.tail(2)


# In[30]:


GOOG.tail(2)


# In[31]:


NFLX.tail()


# <h3>I like Netflix.so,I am going to use Netflix(NFLX) dataset throughout the notebook</h3>

# In[32]:


NFLX.describe()


# In[33]:


NFLX.info()


# <p>There are 253 records and there are no records with null or NA values.</p>
# <h4>We will plot the Adj Close for Netflix over the span of 365 days.</h4>
# <p>Adj Close:The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions.</p>

# In[95]:


NFLX['Adj Close'].plot(legend=True,figsize=(12,6))


# <h4>We will plot Volume for Netflix over 365 days.</h4>
# <p>Volume is the total number of shares of a security that were traded during a particular period of time.</p>

# In[96]:


NFLX['Volume'].plot(legend=True,figsize=(12,6))


# <p>Plot for Volume of stocks tells that there were more number stocks traded between January,2021 and March,2021.</p>
# <h3>Next,We will find the distribution of Stock prices by using Moving Averages of different sizes over 365 days.</h3>
# <p>Moving Average is formed by computing the average of a required quantities over a specific number of periods.</p>
# <p>Here we will use Window sizes of 10,100,200 for calculating Moving Averages.</p>

# In[47]:


move_size=['10','100','200']
for size in move_size:
    NFLX['MA for '+size+' Days']=NFLX['Adj Close'].rolling(window=int(size),center=False).mean()


# In[48]:


NFLX.tail()


# <h4>Great, Now Let's plot it.</h4>

# In[53]:


NFLX[['Adj Close','MA for 10 Days','MA for 100 Days','MA for 200 Days']].plot(subplots=False,figsize=(12,6))


# <p>By noticing the above graph,We are not able to get any useful information.But, We can see that graphs for moving averages over large days are Smoother.</p>
# <h4>Okay,Now lets jump into analysing the Daily Returns of Netflix.</h4>
# <p>The Daily Return measures the dollar change in a stock's price as a percentage of the previous day's closing price. A positive return means the stock has grown in value, while a negative return means it has lost value.</p>

# In[54]:


NFLX['Returns']=NFLX['Adj Close'].pct_change()


# In[57]:


NFLX['Returns'].tail()


# In[59]:


NFLX['Returns'].plot(legend=True,figsize=(12,6),marker='*')


# In[64]:


sns.histplot(NFLX['Returns'].dropna(),bins=50,color='green')


# <p>We can see that the daily returns are distributed similarly on both positive and negative sides.</p>
# <h3>Good,Now we will try to get the correlation between daily returns of differnt stocks.</h3>

# In[66]:


Daily_close = DataReader(companies,'yahoo',start_time,end_time)['Adj Close']


# In[67]:


Daily_close.tail()


# In[68]:


return_pct=Daily_close.pct_change()


# In[69]:


return_pct.tail()


# <h3>We will visualize the correlation between Google and Netflix,Tesla and find any trend available....</h3>

# In[71]:


sns.jointplot(x='GOOG',y='NFLX',data=return_pct,kind='scatter')


# In[72]:


sns.jointplot(x='GOOG',y='TSLA',data=return_pct,kind='scatter')


# <p>It looks like these pairs have poor correlation</p>
# <p>Okay,Let's visualize if there is any relation between all pairs of companies.</p>

# In[74]:


sns.pairplot(return_pct)


# <h4>Looks pretty messy,right.</h4>
# <h3>The solution for this is to use Heatmap.Let's do it.</h3>

# In[76]:


sns.heatmap(return_pct.corr(),annot=True)


# <h4>It seems like there is a higher value of correlation between Google and Netflix(0.46) and Google and Tesla(0.36).</h4>
# <h3>Great,Now let's calculate the Value of Risk for all the stocks.</h3>
# <p>Value at risk estimates how much a set of investments might lose (with a given probability), given normal market conditions, in a time period.</p>
# <p>We can calculate and compare it by comparing standard deviations of stocks for all companies.</p>
# <br>
# <p>Why standard deviation ?</p>
# <p>Well,standard deviation is the range or spread of data points from mean value.Hence if any company has higher standard deviation,it states that the stock values move wildly from low to high and viceversa.</p> 
# <p>Hence, Stocks with lower standard deviation has lower Value of Risk.</p>

# In[77]:


return_pct.dropna().describe()


# In[161]:


rets=return_pct.dropna().describe()
def get_std(comp):
    return round(rets[comp[0]][2],4)*1000
companies['std_val']=companies.apply(get_std,axis=1)
companies.columns=['company','std_val']
companies.head()


# In[162]:


sns.barplot(x='company',y='std_val',data=companies)


# <p>Well,we see that Walmart, Google, Netflix has lower Value of Risk(standard_deviation).Hence it will be safe to invest them,because they has relatively lower probability for loss in stock market.</p> 
# <p>Okay, now we have decided where to invest, but before that we want to know what are the exact loss values for each company's stock.</p>
# <h4>For that,We have to find Value at Risk.And We will also try to predict stock behaviour in future for a given starting price.</h4>
# <p>We can use Bootstrap method and quantile operations to get the possible loss value.</p>
# <p>And we use Monte-Carlo method to predict the future stock behaviour.</p>

# In[215]:


sns.histplot(NFLX['Returns'].dropna(),bins=50)


# In[220]:


return_pct.dropna().head()


# In[221]:


return_pct.dropna()['NFLX'].quantile(0.05)


# <p>Great,Let's infer the above line,<br>Since,we have used 5% in quantile,it tells by 95% confidence,that the worst percentage of loss is 3.4% of the investment.</p>
# <h4>Let's define a function that returns the predictions using <a href="https://www.investopedia.com/terms/m/montecarlosimulation.asp">Monte-Carlo</a> method.</h4>
# <p>This method returns simulations for predicting the range of values over a particular time period.For more info follow above link.</p>

# In[260]:


def monte_carlo(company,name):
    pred_price=np.zeros(365)
    delta_t=1/365
    mean_val=return_pct.dropna().mean()[name]
    stan_val=return_pct.dropna().std()[name]
    shock_val=np.zeros(365)
    drift_val=np.zeros(365)
    initial_price=company['Open'][0]
    pred_price[0]=initial_price
    for i in range(1,365):
        shock_val[i]=np.random.normal(loc=mean_val*delta_t,scale=stan_val*np.sqrt(delta_t))
        drift_val[i]=mean_val*delta_t
        pred_price[i]=pred_price[i-1]+(pred_price[i-1]*(shock_val[i]+drift_val[i]))
    return pred_price


# <h4>Since we have previously determined that Value of Risk is low for Walmart,Google and Netflix stocks,Let's analyze them.</h4>
# <p>For visualizing the working of Monte-Carlo method,we plot 100 simulations.</p>

# In[262]:


for epoch in range(100):
    plt.plot(monte_carlo(GOOG,'GOOG'))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Analysis for Google')


# In[263]:


for epoch in range(100):
    plt.plot(monte_carlo(WMT,'WMT'))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Analysis for Walmart')


# In[264]:


for epoch in range(100):
    plt.plot(monte_carlo(NFLX,'NFLX'))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Analysis for Netflix')


# <p>Okay,The above three visuals show the range of prices after 365 days for a given starting price.</p>
# <p>Usually 100 simulations are not enough for getting better and accurate predictions,<br> so will run the simulations for 15000 times for good predictions(because we don't want to risk it).</p>

# In[266]:


ep=15000
simulation_google=np.zeros(ep)
for current in range(ep):
    simulation_google[current] = monte_carlo(GOOG,'GOOG')[364]


# In[267]:


simulation_walmart=np.zeros(ep)
for current in range(ep):
    simulation_walmart[current] = monte_carlo(WMT,'WMT')[364]


# In[268]:


simulation_netflix=np.zeros(ep)
for current in range(ep):
    simulation_netflix[current] = monte_carlo(NFLX,'NFLX')[364]


# In[270]:


q_google=np.percentile(simulation_google,1)
q_walmart=np.percentile(simulation_walmart,1)
q_netflix=np.percentile(simulation_netflix,1)


# In[292]:


def plot_hist(company,simulation,name):
    quant_val=np.percentile(simulation,1)
    plt.hist(simulation,bins=50)
    plt.figtext(0.6,0.8,s="Start price: $%.2f" %company['Open'][0])
    plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulation.mean())
    plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (company['Open'][0] -quant_val,))
    plt.figtext(0.15,0.6, "q(0.99): $%.2f" % quant_val)
    plt.axvline(x=quant_val, linewidth=4, color='r')
    titl="Final price distribution for "+name+" Stock after 365 days"
    plt.title(titl,weight='bold')


# <h3>Great,</h3>
# <p>Now,By using the above simulations lets plot the distributions of pedicted prices and get the Value of Stock at Risk by using a confidence level of 99%.</p>

# In[293]:


plot_hist(GOOG,simulation_google,'Google')


# <p>From above graph,We can infer that Google stocks has a given start price of USD1408 and Value at Risk is USD39.60.<br>
#     And the red line indicates the probable Value of Stock(USD1368.4) at Risk by using the confidence level of 99%.</p>

# In[294]:


plot_hist(WMT,simulation_walmart,'Walmart')


# <p>From above graph,We can infer that Walmart stocks has a given start price of USD124 and Value at Risk is 2.42USD.<br>
#     And the red line indicates the probable Value of Stock at Risk(USD122.47) by using the confidence level of 99%.</p>

# In[295]:


plot_hist(NFLX,simulation_netflix,'Netflix')


# <p>From above graph,We can infer that Netflix stocks has a given start price of USD448.5 and Value at Risk is USD20.5.<br>
#     And the red line indicates the probable Value of Stock at Risk(USD428) by using the confidence level of 99%.</p>

# <h3>Conclusion:<br><br>
#     Though all the stocks have good range of return values,<br>
#     but by analyzing the visuals and trends, we can conclude that the Stock values for Google,Walmart and Netflix are stable and has lower Value of Risk.</h3>
#     <br>

# <h2>Reported by ANANTHA SAI AVADHANAM</h2>
