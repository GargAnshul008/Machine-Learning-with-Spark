#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: unsupervised model training

Usage:

    $ spark-submit unsupervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, Tokenizer

def main(spark, data_file, model_file):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''
    DF = spark.read.parquet(data_file)
    DF=DF.select("mfcc_00", "mfcc_01", "mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19")
    assembler = VectorAssembler(
    inputCols=["mfcc_00", "mfcc_01", "mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19"],
    outputCol="features")
    #DF = assembler.transform(DF)
    scaler = StandardScaler(inputCol="features", outputCol="features_scaled", withStd=True, withMean=False)
    #scalerModel = scaler.fit(DF)
    #DFnew = scalerModel.transform(DF)
    kmeans = KMeans().setK(100).setSeed(1)
    ###
    # TODO: Enter your code
    ###
        
    pipeline = Pipeline(stages=[assembler,scaler,kmeans])
    model=pipeline.fit(DF)
    model.save(model_file)
   
    pass


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('unsupervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
