

import pandas as pd
from scipy import stats


# loading the data
cars = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Probability distribution & CLT & confidence interval/probability dataset/probability dataset/Cars.csv")

cars.MPG.mean()  # 34.422075728024666
cars.MPG.std()   #  9.131444731795982

# Q1)a.	P(MPG>38) 

z_38 = (38 - cars.MPG.mean())/cars.MPG.std() # (38 - 34.422075728024666)/9.131444731795982
 
z_38   # 0.3918
P_38 = stats.norm.cdf(38, cars.MPG.mean(), cars.MPG.std()) 
P_38   # 0.6524060748417295

# b.	P(MPG<40)
z_40 = (40 - cars.MPG.mean())/cars.MPG.std() 
z_40   # 0.6109
P_40 = stats.norm.cdf(40, cars.MPG.mean(), cars.MPG.std())
P_40   # 0.7293498762151616

# c.	P (20<MPG<50)
z_50 = (50 - cars.MPG.mean())/cars.MPG.std()
z_50   # 1.7059
P_50 = stats.norm.cdf(50, cars.MPG.mean(), cars.MPG.std()) 
P_50  # 0.955992693289364

z_20 = (20 - cars.MPG.mean())/cars.MPG.std() 
z_20  # -1.5794
P_20 = stats.norm.cdf(20, cars.MPG.mean(), cars.MPG.std())
P_20  # 0.05712377632115936

P20_50 = stats.norm.cdf(50, cars.MPG.mean(), cars.MPG.std()) - stats.norm.cdf(20, cars.MPG.mean(), cars.MPG.std())
P20_50   # 0.8988689169682046

P_50 - P_20 #0.8988689169682046
