from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from datetime import datetime
from pyspark.sql.functions import col,udf
from pyspark.sql.types import  (StructType, 
                                StructField, 
                                DateType, 
                                BooleanType,
                                DoubleType,
                                IntegerType,
                                StringType,
                               TimestampType)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
import scipy.stats
from pyspark.sql import functions as sf
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#from pyspark.sql.functions import year, month, dayofmonth

print("Starting")

#Otherwise you will not be able to see any print statements you include
# sc.setLogLevel("ERROR")

sc = SparkContext()
sqlcontext = SQLContext(sc)

root = 'hdfs://wolf.iems.private/user/ttandon'
path = '{root}/HW3/crime/Crimes_-_2001_to_present.csv'.format(root=root)

crime = sqlcontext.read.csv(path, header = True)
crimes = crime
crime.registerTempTable('crime')
print("File Read")


# Change datetime to datetime type
myfunc = udf( lambda x: datetime.strptime( x, '%m/%d/%Y %I:%M:%S %p'), TimestampType())
df = crime.withColumn( 'Date_time', myfunc(col('Date'))).drop("Date")
#df.select(df["Date_time"]).show(5)
df1 = df.withColumn( "Date_month", month(df.Date_time))
df1 = df1.withColumn( "Date_year", year(df.Date_time))
f = df1.select("Date_month", "Date_year")
gg = df1.groupby("Date_month", "Date_year").count()

crime_group_mean = gg.groupby('Date_month').mean('count').orderBy('Date_month')
#crime_group_mean.show()

l1 = crime_group_mean.select('avg(count)').collect()
l2 = crime_group_mean.select('Date_month').collect()

x = []
y = []

for i in l1:
    temp1 = i.asDict()
    x.append(temp1['avg(count)'])
  
    
for j in l2:
    temp2 = j.asDict()
    y.append(temp2['Date_month'])

plt.bar(y,x)
plt.savefig("hist.png")

# end of Q1 -----------------------------------------------------------------------------------------------------------------------  


#Q4 

df1 = df1.withColumn( "Date_day", dayofmonth(df.Date_time))
df1 = df1.withColumn( "Date_hour", hour(df.Date_time))
df1 = df1.withColumn( "Date_week", date_format((df.Date_time),'EEEE' ))
df1 = df1.withColumn( "Date_date", sf.concat(sf.col("Date_month"),sf.lit("/"),sf.col("Date_day"),sf.lit("/"), sf.col("Date_year")))

crime_per_month = df1.groupby("Date_month").count().orderBy("Date_month").toPandas()
crime_per_month.to_csv('crime_per_month.csv', index = False)

#by day 
crime_per_weekday = df1.groupby("Date_week").count().orderBy("count").toPandas()
crime_per_weekday.to_csv('crime_per_weekday.csv', index = False)

#by hour
crime_per_hour = df1.groupby("Date_hour").count().orderBy("Date_hour").toPandas()
crime_per_hour.to_csv('crime_per_hour.csv', index = False)


#end of Q4 ----------------------------------------------------------------------------------------------------------------------- 


#2.1 

crime_top_ten = crime.filter(crime.Year >= 2013 ).groupBy("Block").count().orderBy("count", ascending = False).limit(10).toPandas()
crime_top_ten.to_csv('crime_top_ten.csv', index = False)
#end of 2.1


#2.2 

crime_beat = df1.filter(df1.Year >= 2011 ).groupby("Beat", "Date_date").count()
crime_beat.registerTempTable("crime_beat")
crime_beat_p =crime_beat.coalesce(1).toPandas()

temp221 = crime_beat_p.pivot( index = 'Date_date', columns = 'Beat') ['count'].fillna(0)
temp222 = temp221.corr()
temp223 = temp222.unstack().sort_values(kind="quicksort", ascending = False)
temp224 = temp223[ 302:]

temp224.to_csv('Correlated_beat.csv')
#end of 2.2


# 2.3 

# By year 

crime = df1
emanuel = crime.filter(crime.Date_year > 2011).groupby("Year").count().toPandas()
daly = crime.filter(crime.Date_year <= 2011).groupby("Year").count().toPandas()
t1 = scipy.stats.ttest_ind(emanuel['count'], daly['count'], equal_var=False)
np.savetxt( "Ttest_byyear.txt", t1, header = "Ttest_indResult(statistic ; pvalue)") 

# By beat and month 
emanuel = crime.filter(crime.Date_year > 2011).groupby("Beat", "Date_month").count().toPandas()
daly = crime.filter(crime.Date_year <= 2011).groupby("Beat", "Date_month").count().toPandas()
t2 = scipy.stats.ttest_ind(emanuel['count'], daly['count'], equal_var=False)
np.savetxt( "Ttest_bybeatandmonth.txt", t2, header = "Ttest_indResult(statistic ; pvalue)") 

# end of 2.3 

# end of Q2----------------------------------------------------------------------------------------------------------------------- 

# Q3

crimes.registerTempTable('crimes')
crime_date_format = crimes.withColumn('Date',to_date("Date", "MM/dd/yyyy HH:mm:ss"))


# create concatenated week and year
crime_week = crime_date_format.withColumn('Week', weekofyear('Date'))
crime_week = crime_week.withColumn('Week', col('Week').cast('string'))
crime_week.registerTempTable('crime_week')
crime_week = sqlcontext.sql("SELECT *,case when length(Week) == 1 then concat('0',Week) else Week end as updated_week FROM crime_week")
crime_week_year = crime_week.withColumn('Year_Week', concat(col('Year'), col('updated_week')))
crime_week_year.registerTempTable('crime_week_year')

# create arrest and domestic columns
crime_year_week = crime_week_year.groupBy("Beat", "Year_Week").agg(count("ID").alias("label"))
crime_arrest = crime_week_year.withColumn('Arrest', crimes['Arrest'].cast('boolean').cast('integer')).groupBy("Beat","Year_Week").agg(sum("Arrest").alias("Arrest"))
crime_domestic = crime_week_year.withColumn('Domestic', crimes['Domestic'].cast('boolean').cast('integer')).groupBy("Beat","Year_Week").agg(sum("Domestic").alias("Domestic"))
crime_beat_week = crime_year_week.join(crime_arrest, on=['Beat', 'Year_Week']).join(crime_domestic, on = ['Beat','Year_Week'])
crime_beat_week = crime_beat_week.withColumn('week', substring('Year_Week', 5, 2).cast('integer'))
crime_beat_week.orderBy('Beat', 'Year_Week')
crime_beat_week.registerTempTable('crime_beat_week')

# create lagged features
crime_beat_week_lagged = (crime_beat_week.withColumn('Lag1', lag(col('label'), count = 1).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged = (crime_beat_week_lagged.withColumn('Lag2', lag(col('label'), count = 2).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged = (crime_beat_week_lagged.withColumn('Lag3', lag(col('label'), count = 3).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged = (crime_beat_week_lagged.withColumn('Lag4', lag(col('label'), count = 4).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged = (crime_beat_week_lagged.withColumn('Lag_Arrest', lag(col('Arrest'), count = 1).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged = (crime_beat_week_lagged.withColumn('Lag_Domestic', lag(col('Domestic'), count = 1).over(Window().partitionBy('Beat').orderBy('Year_Week'))).na.drop())
crime_beat_week_lagged.registerTempTable('crime_beat_week_lagged')

#export_data.to_csv("q2_3_beat_cnt.csv", index=False)
# run model
input_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag_Arrest', 'Lag_Domestic', 'week']
assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
rf = RandomForestRegressor(numTrees=30)
stages = [assembler, rf]
pipeline = Pipeline(stages=stages)
param_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 6, 7]).build()
evaluator = RegressionEvaluator(labelCol='label',
                                predictionCol='prediction',
                                metricName='r2')

rf_grid = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
).fit(crime_beat_week_lagged)
lagged_fitted = rf_grid.transform(crime_beat_week_lagged)
# print r-square

eval_metric = evaluator.evaluate(lagged_fitted)



print( "evaluation answer")
print( "-----------------------------------")
print(eval_metric)
print( "-----------------------------------")

# end of Q3 -----------------------------------------------------------------------------------------------------------------------  




