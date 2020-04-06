#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
# TODO: you may need to add imports here


def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''
    DF1= spark.read.parquet(data_file)
    DF = DF1.sample(False, 0.1)
    print(DF.count())
    DF=DF.select("genre","mfcc_00", "mfcc_01", "mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19")
    assembler = VectorAssembler(
    inputCol=["mfcc_00", "mfcc_01", "mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19"],
    outputCol="features1")
    
    #DF = assembler.transform(DF)
    scaler = StandardScaler(inputCol="features1", outputCol="features", withStd=True, withMean=False)
    #scalerModel = scaler.fit(DF)
    #DFnew = scalerModel.transform(DF)
    indexer = StringIndexer(inputCol="genre", outputCol="label",handleInvalid="keep")
    #DF1 = indexer.fit(DFnew).transform(DFnew)
    mlor = LogisticRegression(maxIter=10 ,family="multinomial")
    pipeline = Pipeline(stages=[assembler,scaler,indexer,mlor])
    paramGrid = ParamGridBuilder() \
                .addGrid(mlor.elasticNetParam,[1.0,0.8,0.4,0.2,0.6]) \
                .addGrid(mlor.regParam, [0.1, 0.2,0.3,0.01,0.001])\
                .build()
   
    
    ###
    # TODO: YOUR CODE GOES HERE
    ###
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
    cvModel = crossval.fit(DF)
    cvBest=cvModel.bestModel
    cvBest.save(model_file)
    
    pass




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
