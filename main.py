# Import libraries
from pyspark.mllib.recommendation import *
import random
from operator import *
from collections import defaultdict

#import libraries
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession ,Row
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType,StructField,IntegerType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

appName="Collaborative Filtering with PySpark"

#initialize the spark session
spark = SparkSession.builder.appName(appName).getOrCreate()

#get sparkcontext from the sparksession
sc = spark.sparkContext
sqlContext = SQLContext(sc)
path = "train_triplets.txt"
data = spark.read.csv(path, sep=r'\t', header=False)
data = data.withColumnRenamed("_c0","user").withColumnRenamed("_c1","song").withColumnRenamed("_c2","playCount")

print("Schema:  ")
data.printSchema()


import pyspark.sql.functions as F
df2= data.withColumn("userId", F.hash(col("user")))

df2= df2.withColumn("songId", F.hash(col("song")))
# userId_change = data.select('user').distinct().select('user', F.monotonically_increasing_id().alias('userid'))
# songId_change = data.select('song').distinct().select('song', F.monotonically_increasing_id().alias('songid'))
# df2 = data.join(userId_change, 'user').join(songId_change, 'song')
df2 = df2.withColumn("playCount",col("playCount").cast(IntegerType()))
df2.show()
df2 = df2.limit(100000)

(training, test) = df2.randomSplit([0.7, 0.3],seed = 20)

def makeSongRecommendation(numberOfRecommendation = 10):
    # ___ joining track details
    path = "unique_tracks.txt"
    tracks = spark.read.csv(path, sep="<SEP>")
    tracks.show()
    tracks = tracks.withColumnRenamed("_c0", "tid").withColumnRenamed("_c1", "song").withColumnRenamed("_c2",
                                                                                                       "ArtistName").withColumnRenamed(
        "_c3", "SongTitle")
    tracks = tracks.withColumn("songId", F.hash(col("song")))

    temp = df2.groupby("song").avg("playCount")
    temp2 = df2.join(temp, on="song")
    temp2.limit(10).show()
    temp2 = temp2.withColumn("rating", F.col("playCount") / F.col("avg(playCount)"))

    # ________ the model block

    (training, test) = temp2.randomSplit([0.7, 0.3], seed=20)
    # rank=20,maxIter=10, regParam=0.01,

    als = ALS(rank=100, maxIter=10, regParam=0.15,
              userCol="userId", itemCol="songId", ratingCol="rating",
              coldStartStrategy="drop",
              implicitPrefs=True)
    best_model = als.fit(training)

    # View the predictions
    test_predictions = best_model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="playCount",
        predictionCol="prediction")
    RMSE = evaluator.evaluate(test_predictions)
    print(RMSE)

    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(numberOfRecommendation)
    # recommendations.show()

    # recommend for user table
    from pyspark.sql.functions import explode
    nrecommendations = recommendations \
        .withColumn("rec_exp", explode("recommendations")) \
        .select('userId', col("rec_exp.songId"), col("rec_exp.rating"))
    nrecommendations.limit(numberOfRecommendation).show()
    return [nrecommendations,temp2,tracks]



# show output
def compareOutputsOfSongRecommendation(nrecommendations,temp2,userId,tracks):
    fil = "userId = "+str(userId)
    recommedationOutput = nrecommendations.join(tracks, on='songId').filter(fil).sort('rating',ascending=False).select("songId","userId","rating","songTitle",)
    actualUserSongs = temp2.join(tracks,on="songId").filter(fil).sort('playCount', ascending=False).select("songId","userId","playCount","songTitle",).limit(10)
    return [recommedationOutput,actualUserSongs]

def makeArtistRecommendationModel(tracks,numOfRecommendations = 5):

    path = "unique_artists.txt"
    artists = spark.read.csv(path, sep="<SEP>")
    artists = artists.withColumnRenamed("_c0", "artistId").withColumnRenamed("_c1", "AmbId").withColumnRenamed("_c2","tid").withColumnRenamed("_c3", "ArtistName")
    artists = artists.withColumn("aId", F.hash("artistId"))
    temp3 = artists.select("aId", "ArtistId", "tid").join(tracks, on="tid")
    temp3 = temp3.drop("songId")
    temp3 = temp3.join(df2, on="song").select("song", "ArtistId", "userId", "playCount", "ArtistName")
    temp3 = temp3.groupby(["ArtistId", "userId"]).sum("playCount").sort("sum(playCount)", ascending=False)
    temp3 = temp3.withColumn("aId", F.hash(col("ArtistId")))
    artistRecommendationData = temp3
    (training, test) = artistRecommendationData.randomSplit([0.7, 0.3], seed=20)

    # rank=20,maxIter=10, regParam=0.01,
    als = ALS(rank= 15,maxIter=10,regParam=0.01,
        userCol="userId", itemCol="aId", ratingCol="sum(playCount)",
        coldStartStrategy="drop",
        implicitPrefs=True)
    best_model = als.fit(training)
    # Import the requisite packages

    from pyspark.ml.evaluation import RegressionEvaluator

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="sum(playCount)",
        predictionCol="prediction")

    test_predictions = best_model.transform(test)
    RMSE = evaluator.evaluate(test_predictions)
    print("RMSE: ",RMSE)
    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(numOfRecommendations)
    from pyspark.sql.functions import explode
    nrecommendations = recommendations \
        .withColumn("rec_exp", explode("recommendations")) \
        .select('userId', col("rec_exp.aId"), col("rec_exp.rating"))
    nrecommendations.limit(numOfRecommendations).show()
    return [artists,artistRecommendationData,nrecommendations]

def CompareArtistRecommendation(userID,artists,artistRecommendationData,nrecommendations):
    fil = "userId = "+str(userID)
    recommededData = nrecommendations.join(artists.select("aId", "ArtistId", "ArtistName"), on='aId').filter(fil).sort('rating', ascending=False).select("userId", "rating", "ArtistId","ArtistName")
    actualData = artistRecommendationData.join(artists,on="ArtistId").filter(fil).sort('sum(playCount)', ascending=False).select("ArtistId","userId","sum(playCount)","ArtistName",).limit(20)
    return [recommededData,actualData]




#************************** MAIN CALLS ************

songRecommeder, songMasterTable , tracksTable = makeSongRecommendation()
artists , artistRecommendationData , artistRecommeder = makeArtistRecommendationModel(tracksTable)

recData,actualData = CompareArtistRecommendation(-2130942721,artists,artistRecommendationData,artistRecommeder)
recData.show()
actualData.show()

import PySimpleGUI as sg
import numpy as np
sg.theme('DarkTeal9')
def ArtistWindow(userId):
    recData, actualData = CompareArtistRecommendation(userId, artists, artistRecommendationData, artistRecommeder)
    headings1 = list(recData.columns)
    headings2 = list(actualData.columns)
    Table1 = [[sg.Text("Recommendation Systems Recommended Artist") ],[ sg.Table(values=np.array(recData.collect()),headings=headings1,num_rows=10,size=10)]]
    Table2 = [[sg.Text("Actual users Artist Data") ],[sg.Table(values=np.array(actualData.collect()),headings=headings2,num_rows=10,size=10)]]
    layout=[[sg.Column(Table1),sg.VSeparator(),sg.Column(Table2)]]
    window = sg.Window(title="Recommendation System",layout=layout)
    while True:
        event,values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break

    window.close()

def songWindow(userId):
    recData, actualData = compareOutputsOfSongRecommendation(songRecommeder,songMasterTable,userId,tracksTable)
    headings1 = list(recData.columns)
    headings2 = list(actualData.columns)
    Table1 = [[sg.Text("Recommendation Systems Recommended songs") ],[ sg.Table(values=list(recData.collect()),headings=headings1,num_rows=10,auto_size_columns=True)]]
    Table2 = [[sg.Text("Actual users song Data") ],[sg.Table(values=list(actualData.collect()),headings=headings2,num_rows=10,auto_size_columns=True)]]
    layout=[[sg.Column(Table1),sg.VSeparator(),sg.Column(Table2)]]
    window = sg.Window(title="Recommendation System",layout=layout)
    while True:
        event,values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break

    window.close()

def recArt():
    listOfUsers = list(artistRecommeder.select("userId").distinct().collect())
    layout = [[sg.Table(values=listOfUsers,headings=["userId"],num_rows=10,size=10,enable_events=True,key="user")]]
    window = sg.Window(title="Select User",layout=layout)
    while True:
        event,values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break
        if event =="user":
            print(listOfUsers[values["user"][0]][0])
            ArtistWindow(listOfUsers[values["user"][0]][0])

    window.close()

def recSong():
    listOfUsers = list(songRecommeder.select("userId").distinct().collect())
    layout = [[sg.Table(values=listOfUsers, headings=["userId"], num_rows=10, size=10, enable_events=True, key="userid")]]
    window = sg.Window(title="Select User", layout=layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break
        if event == "userid":
            print(listOfUsers[values["userid"][0]][0])
            songWindow(listOfUsers[values["userid"][0]][0])

    window.close()

def mainWindow():
    window = sg.Window(title="Select User", layout=[[sg.Button("Recommend Songs")],[sg.Button("Recommend Artist")]],size=(300,300),element_justification='c')
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WINDOW_CLOSED:
            break
        if event == "Recommend Songs":
            recSong()
        if event == "Recommend Artist":
            recArt()

    window.close()

mainWindow()