#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training code for Row-rank Matrix Factorisation using ALS-WS explicit feedback(ratings)
   Ref: https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.html
"""

import sys
import os
import argparse
import itertools
from math import sqrt
from operator import add

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


def parseMovie(line):
  """
  Parses a movie record in MovieLens format movieId::movieTitle .
  """
  #fields = line.strip().split("::")
  fields = line.strip().split(",")
  return int(fields[0]), fields[1]


def parseRating(line):
  """
  Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
  """
  #fields = line.strip().split("::")
  fields = line.strip().split(",")
  return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))


def loadRatings(ratingsFile):
  """
  Load ratings from file.
  """
  if not os.path.isfile(ratingsFile):
    print "File %s does not exist." % ratingsFile
    sys.exit(1)
  f = open(ratingsFile, 'r')
  ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
  f.close()
  if not ratings:
    print "No ratings provided."
    sys.exit(1)
  else:
    return ratings


def computeRMSE(model, data, n):
  """
  Compute RMSE (Root Mean Squared Error).
  """
  predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
  predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
    .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
    .values()
  return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--executorMemory', 
                      default='12g', help='executor-memory')
  parser.add_argument('--driverMemory', 
                      default='3g', help='driver-memory')
  parser.add_argument('--moveLensDataDir', 
                      default='hdfs://hdfs:9000/movielens/ml-latest/', help='dir')
  parser.add_argument('--personalRatingsFile', 
                      default='personalRatings.txt', help='personalRatingsFile')
  args = parser.parse_args()
  params = vars(args)

  # set up environment
  conf = SparkConf() \
    .setAppName("MovieLensALS-latest")\
    .set("spark.executor.memory", params['executorMemory']) \
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)
  # FIXME: doesn't need to set checkpoint
  #sc.setCheckpointDir('hdfs://url:9000/checkpoint')
  #ALS.checkpointInterval = 2

  # set home directory for loading ratings and movie titles
  movieLensHomeDir = params['moveLensDataDir']

  # load personal ratings
  myRatingsRDD = sc.textFile(os.path.join(movieLensHomeDir, params['personalRatingsFile']))\
                   .map(parseRating)\
                   .filter(lambda r: r[1][2] > 0)\
                   .map(lambda r: r[1])
  
  # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
  ratings = sc.textFile(os.path.join(movieLensHomeDir, 
                                     "ratings.dat")).map(parseRating)

  # movies is an RDD of (movieId, movieTitle)
  movies = dict(sc.textFile(os.path.join(movieLensHomeDir, 
                                         "movies.dat")).map(parseMovie).collect())

  numRatings = ratings.count()
  numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
  numMovies = ratings.values().map(lambda r: r[1]).distinct().count()
  print "Got %d ratings from %d users on %d movies." % \
          (numRatings, numUsers, numMovies)

  # set numPartitions
  numPartitions = 12
  training = ratings.filter(lambda x: x[0] < 6)\
                    .values()\
                    .union(myRatingsRDD)\
                    .repartition(numPartitions)\
                    .cache()
  validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
                      .values() \
                      .repartition(numPartitions) \
                      .cache()
  test = ratings.filter(lambda x: x[0] >= 8).values().cache()
  numTraining = training.count()
  numValidation = validation.count()
  numTest = test.count()
  print "Training: %d, validation: %d, test: %d" % \
          (numTraining, numValidation, numTest)

  """
  ranks = [2, 4, 8, 16]
  lambdas = [0.01, 0.5, 0.1, 0.2, 0.4]
  numIters = range(1,13)
  """
  # meta for training
  ranks = [16]
  lambdas = [0.1]
  numIters = [7]
  bestModel = None
  bestValidationRMSE = float("inf")
  bestRank = 0
  bestLambda = -1.0
  bestNumIter = -1
  outputModelFilename = 'hdfs://url:9000/movielens/model/als_explicit'
  for lmbda, rank, numIter in itertools.product(lambdas, ranks, numIters):
    model = ALS.train(training, rank, numIter, lmbda)
    validationRMSE = computeRMSE(model, validation, numValidation)
    trainRMSE = computeRMSE(model, training, numTraining)
    print "trn: %f, val: %f, rank = %d, lambda = %.1f, and numIter = %d." % \
      (trainRMSE, validationRMSE, rank, lmbda, numIter)
    if (validationRMSE < bestValidationRMSE):
      bestModel = model
      bestValidationRMSE = validationRMSE
      bestRank = rank
      bestLambda = lmbda
      bestNumIter = numIter
      bestModel.save(sc, outputModelFilename)

  testRMSE = computeRMSE(bestModel, test, numTest)

  # evaluate the best model on the test set
  # The best model was trained with rank = 16 and lambda = 0.1, and numIter = 7, and its RMSE on the test set is 0.818418.
  print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRMSE)

  # Comparing to a naive baseline
  meanRating = training.union(validation).map(lambda x: x[2]).mean()
  baselineRMSE = sqrt(test.map(lambda x: (meanRating - x[2])**2).reduce(add) / numTest)
  improvement = (baselineRMSE - testRMSE) / baselineRMSE * 100
  print 'The best model improves the baseline by %.2f%%' % improvement

  # clean up
  sc.stop()
