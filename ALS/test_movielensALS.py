#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Testing code for Row-rank Matrix Factorisation using ALS-WS explicit feedback(ratings)
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
                      default='hdfs://url:9000/movielens/ml-latest/', help='dir')
  parser.add_argument('--personalRatingsFile', 
                      default='personalRatings.txt', help='personalRatingsFile')
  args = parser.parse_args()
  params = vars(args)

  # set up environment
  conf = SparkConf() \
    .setAppName("MovieLensALS-latest-test")\
    .set("spark.executor.memory", params['executorMemory']) \
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)

  # load ratings and movie titles
  movieLensHomeDir = params['moveLensDataDir']

  # movies is an RDD of (movieId, movieTitle)
  movies = dict(sc.textFile(os.path.join(movieLensHomeDir, 
                                         "movies.dat")).map(parseMovie).collect())

  inputModelFilename = 'hdfs://url:9000/movielens/model/als_explicit'
  numPartitions = 12

  model = MatrixFactorizationModel.load(sc, inputModelFilename)
  model.userFeatures().repartition(numPartitions).cache()
  model.productFeatures().repartition(numPartitions).cache()

  # ratings prediction
  myRatings = sc.textFile(os.path.join(movieLensHomeDir, params['personalRatingsFile']))\
                .map(parseRating)\
                .filter(lambda r: r[1][2] > 0)\
                .map(lambda r: r[1])\
                .collect()
  myRatedMovieIds = set([x[1] for x in myRatings])

  candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds], 
                              numPartitions)
  predictions = model.predictAll(candidates.map(lambda x: (0, x))).collect()
  recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

  print 'Movies recommended for you:'
  for rank in xrange(len(recommendations)):
    print ("%2d: %s" % (rank + 1, movies[recommendations[rank][1]])).encode('ascii', 'ignore')
  
  # clean up
  sc.stop()
