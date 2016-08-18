#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  driver code for evaluting ranking result
"""

import sys
import os
import argparse

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

import numpy as np
from RankMetric import ndcg_at_k, average_precision
import itertools


def mapToList(entry):
  """convert relevence scores for each nearest-neighbor to list which is used in nDCG"""
  try:
    return (entry[0], entry[1].split(' '))
    #return entry[1].split(' ')
  except Exception as err:
    print('ERROR: %s' % err)
    return None

def mapToBinarizeRel(entry):
  """convert relevence scores to binary scores which is used in mAP."""
  try:
    rel = np.asarray(entry[1].split(' '), dtype=np.float64)
    rel = np.asarray(rel >= 2, dtype=np.float64)
    return (entry[0], rel)
  except Exception as err:
    print('ERROR: %s' % err)
    return None

def runApp(sc, sqlSC, numPartitions=8):
  inputDF = sqlSC.read\
                 .format('jdbc')\
                 .options(url='jdbc:mysql:/url/?user=oops&password=oops',
                          dbtable=u'(SELECT __query_img_url__, __evaluation_score__ FROM evaluation_result_taey16 WHERE verification <> -1 ORDER BY __rank__) inputDF',
                          driver='com.mysql.jdbc.Driver')\
                 .load()\
                 .collect()

  inputRDD = sc.parallelize(inputDF).cache()
  totalSamples = inputRDD.count()
  print 'Finished parallelize %d' % totalSamples
  print '# of Partitions: %d' % inputRDD.getNumPartitions()

  # get query, nearest-neightor pairs
  inputRDD = inputRDD.reduceByKey(lambda v1,v2: \
                                  '%s %s' % (str(v1), str(v2)))

  # start evaluation
  atK = range(1,6)
  metricFlag = [False, True]
  scores, score = [], []
  for useMAP, k in itertools.product(metricFlag, atK):
    print('@k: %d, useMAP: %d' % (k, useMAP))

    if useMAP:
      rankRDD = inputRDD.map(mapToBinarizeRel)\
                        .filter(lambda x: x <> None)
      print(rankRDD.collect())
      metricRDD = rankRDD.map(lambda x: average_precision(x, k))
    else:
      rankRDD = inputRDD.map(mapToList)\
                        .filter(lambda x: x <> None)
      print(rankRDD.collect())
      metricRDD = rankRDD.map(lambda x: ndcg_at_k(x,k))

    score.append(metricRDD.values().mean())
    """
    for item in rankRDD.collect():
      if useMAP: score = average_precision(item)
      else: score = ndcg_at_k(item, k)
      print(score)
    """
  print(score)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--executorMemory', 
                      default='6G', help='executor-memory')
  parser.add_argument('--driverMemory', 
                      default='3G', help='driver-memory')
  parser.add_argument('--numPartitions', type=int, 
                      default=8, help='driver-memory')
  args = parser.parse_args()
  params = vars(args)

  conf = SparkConf() \
    .setAppName("executor_match")\
    .set("spark.executor.memory", params['executorMemory']) \
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)
  sqlSC = SQLContext(sc)

  runApp(sc, sqlSC, params['numPartitions'])

  sc.stop()

