#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Full-pair exact nearest neighbor matching code
"""

import sys
import os
import argparse

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg.distributed import *

import urllib
import urllib2
import json
import random

import numpy as np
from scipy import spatial

reload(sys)
sys.setdefaultencoding('utf-8')

headers = {'User-Agent': 
           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/602.1.25 (KHTML, like Gecko) Version/9.1.1 Safari/601.6.10'}

requestPrefix = \
  'http://%s:%d/request_handler?url=http://%s&%s&is_browser=0'

daemonConfig = {'host': ['url', 'url'],
                'port': [8081, 7081]}

signatureDim = 832 * 9 / 64 + 1024 / 64
colorFeatureDim = 18 * 3 * 3 + 4
def sanityCheck(item, requestURL):
  global signatureDim
  global colorFeatureDim
  successFlag = True

  if item['result_roi'] == False:
    print('ERROR: in result[result_roi], req: %s' % requestURL)
    successFlag = False
  elif item['result_sentence'] == False:
    print('ERROR: in result[result_sentence], req: %s' % requestURL)
    successFlag = False
  elif item['roi'] == {} or item['roi'] == None:
    print('ERROR: in result[roi], req: %s' % requestURL)
    successFlag = False
  elif item['sentence'] == {} or item['sentence'] == None:
    print('ERROR: in result[sentence], req: %s' % requestURL)
    successFlag = False
  elif item['result_signature'] == False:
    print('ERROR: in result[result_signature], req: %s' % requestURL)
    successFlag = False
  elif item['result_color'] == False:
    print('ERROR: in result[result_color], req: %s' % requestURL)
    successFlag = False
  elif item['signature'] == {} or item['signature'] == None:
    print('ERROR: in result[signature], req: %s' % requestURL)
    successFlag = False
  elif item['color_feature'] == {} or item['color_feature'] == None:
    print('ERROR: in result[color_feature], req: %s' % requestURL)
    successFlag = False
  elif item['lctgr_nm'] == '' or item['lctgr_nm'] == None:
    print('ERROR: in result[lctgr_nm], req: %s' % requestURL)
    successFlag = False
  elif item['mctgr_nm'] == '' or item['mctgr_nm'] == None:
    print('ERROR: in result[mctgr_nm], req: %s' % requestURL)
    successFlag = False
  elif item['sctgr_nm'] == '' or item['sctgr_nm'] == None:
    print('ERROR: in result[sctgr_nm], req: %s' % requestURL)
    successFlag = False

  if item['result_signature'] <> False:
    for cate_id, signature in item['signature'].iteritems():
      if len(signature[0]) <> signatureDim:
        print('ERROR: in result[signature], signatureDim mismatched!!, req: %s' % requestURL)
        successFlag = False
      break
  else: 
    successFlag = False
  if item['result_color'] <> False:
    for cate_id, color_feature in item['color_feature'].iteritems():
      if len(color_feature[0]) <> colorFeatureDim:
        print('ERROR: in result[color_feature], colorFeatureDim mismatched!! req: %s' % requestURL)
        successFlag = False
      break
  else: 
    successFlag = False

  return successFlag


def process(entry):
  reload(sys)
  sys.setdefaultencoding('utf-8')

  global daemonConfig
  global headers
  global requestPrefix

  port = daemonConfig['port'][random.randint(0, len(daemonConfig['port'])-1)]
  host = daemonConfig['host'][random.randint(0, len(daemonConfig['host'])-1)]
  #print(entry['__org_img_url__'])

  try:
    imageURL = entry[0]
    apiParams = urllib.urlencode({'lctgr_nm': entry[1], 
                                  'mctgr_nm': entry[2], 
                                  'sctgr_nm': entry[3]})
    requestURL = requestPrefix % (host, port, imageURL, apiParams)
    print(requestURL)

    opener = urllib2.build_opener()
    opener.addheaders = headers.items()
    response = opener.open(requestURL) 
    items = json.loads(response.read())

    successFlag = sanityCheck(items, requestURL)
    if successFlag:
      outputDict = {'url': items['url'], 
                    'roi': items['roi'].values()[0][0], 
                    'feature': items['feature'].values()[0][0],
                    'color_feature': items['color_feature'].values()[0][0],
                    'sentence': items['sentence'][0]}
      return json.dumps(outputDict, separators=(',', ':'))
      #return outputDict
  except Exception as err:
    print('ERROR: %s' % err)


def getFeature(item):
  return [item['url'], np.concatenate((item['feature'], 
                                       item['color_feature']), 
                                       axis=0)]


def runApp(sc, sqlSC, daemonConfig, numPartitions=8):
  """
  # code-fragment for calling demon_11st
  inputDF = sqlSC.read\
                 .format('jdbc')\
                 .options(url='jdbc:mysql://url:3306/dbname?user=oops&password=oops',
                          dbtable=u'(select __org_img_url__, __lctgr_nm__, __mctgr_nm__, __sctgr_nm__ from __7M limit 800000) df',
                          driver='com.mysql.jdbc.Driver')\
                 .load()\
                 .collect()

  inputRDD = sc.parallelize(inputDF, numPartitions).cache()
  totalSamples = inputRDD.count()
  print 'Finished parallelize %d' % totalSamples
  print '# of Partitions: %d' % inputRDD.getNumPartitions()

  outputRDD = inputRDD.map(process)\
                      .filter(lambda x: x <> None)
  outputRDD.saveAsTextFile('hdfs://url:9000/11st_fashion_all_7M_feature')
  #outputRDD.toDF().write.format('json').save('hdfs://url:9000/11st_fashion_all_7M_feature')
  print 'Feature Extraction Finished...'
  """

  # code-fragment for exact NN matching

  # load json files
  #inputJSONFilePath = 'hdfs://url:9000/all_7M_feature/'
  inputJSONFilePath = 'hdfs://url:9000/all_240K_feature/'
  inputJSONRDD = sc.textFile(inputJSONFilePath).map(json.loads)

  # extract feature-vector only
  featureRDD = inputJSONRDD.map(getFeature).cache()
  # exact nearest neighbor matching
  # Ref: http://rnowling.github.io/software/engineering/2016/03/18/optimizing-duplicate-document-detection-in-apache-spark.html
  distanceRDD = featureRDD.cartesian(featureRDD)\
                          .filter(lambda (x,y): x[0] < y[0])\
                          .map(lambda ((query,q_feat),(ref,ref_feat)): \
                               (query,(spatial.distance.cosine(q_feat,ref_feat),ref)))\
                          .reduceByKey(lambda v1,v2: v1+v2)
  print(distanceRDD.takeSample(True, 1)[0])


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

  runApp(sc, sqlSC, daemonConfig, params['numPartitions'])

  sc.stop()
