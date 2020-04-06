#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: supervised model testing

Usage:

    $ spark-submit supervised_test.py hdfs:/path/to/load/model.parquet hdfs:/path/to/file

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
# TODO: you may need to add imports here


def main(spark, model_file, data_file):
    DF = spark.read.parquet(data_file)
    DF.select("genre","mfcc_00", "mfcc_01", "mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19")
    model=PipelineModel.load(model_file)
    predictions = model.transform(DF)
    predictionAndLabels = predictions.select(['label', 'prediction']) \
                            .rdd.map(lambda line: (line[1], line[0]))
        metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''
    # Classes Statistics
    labels = predictions.rdd.map(lambda lp: lp.label).distinct().collect()
    #printing(labels)
    for label in sorted(labels):
       print("Class %s precision = %s" % (label, metrics.precision(label)))
       print("Class %s recall = %s" % (label, metrics.recall(label)))
       print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

    ###
    # TODO: YOUR CODE GOES HERE
    ###
    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
    
   
    pass




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_test').getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
