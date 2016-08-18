#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Dirver code for call demon_11st
"""

import sys
import os
import argparse

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

import urllib
import urllib2
import json
import random
reload(sys)
sys.setdefaultencoding('utf-8')

headers = {'User-Agent': 
           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/602.1.25 (KHTML, like Gecko) Version/9.1.1 Safari/601.6.10'}

requestPrefix = \
  'http://%s:%d/request_handler?url=http://%s&%s&is_browser=0'

daemonConfig = {'host': ['10.202.34.211', '10.202.35.109'],
                'port': [8081, 7081]}

# FIXME: take care of # of dimension for each feature-vector
signatureDim = 832 * 9 / 64 + 1024 / 64
colorFeatureDim = 18 * 3 * 3 + 4
def sanityCheck(item, requestURL):
  """Sanity check for response-json file"""

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
  """mapper for each entry"""

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
                    'roi': items['roi'], 
                    'signature': items['signature'],
                    'feature_color': items['color_feature'],
                    'sentence': items['sentence']}
      return json.dumps(outputDict, separators=(',', ':'))
  except Exception as err:
    print('ERROR: %s' % err)
    return None


def runApp(sc, sqlSC, daemonConfig, numPartitions=8):
  inputDF = sqlSC.read\
                 .format('jdbc')\
                 .options(url='jdbc:mysql://url/?user=oops&password=oops',
                          dbtable=u'(select __org_img_url__, __lctgr_nm__, __mctgr_nm__, __sctgr_nm__ from 11st_fashion_all_7M limit 800) inputDF',
                          driver='com.mysql.jdbc.Driver')\
                 .load()\
                 .collect()

  # convert DataFraem to RDD
  inputRDD = sc.parallelize(inputDF, numPartitions).cache()
  totalSamples = inputRDD.count()
  print 'Finished parallelize %d' % totalSamples
  print '# of Partitions: %d' % inputRDD.getNumPartitions()

  # call demon_11st and then filtering None of response
  outputRDD = inputRDD.map(process)\
                      .filter(lambda x: x <> None)
  # Save json dump into hdfs
  outputRDD.saveAsTextFile('hdfs://oops:9000/11st_fashion_all_7M_feature')
  print 'Finished feature extraction'
  # FIXME: calling outputRDD.count() may lead to much time!!
  #processedSamples = outputRDD.count()
  #print 'Finished %d, missed: %d' % (processedSamples, totalSamples - processedSamples)


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

  conf = SparkConf()\
    .setAppName("executor_detect")\
    .set("spark.executor.memory", params['executorMemory'])\
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)
  sqlSC = SQLContext(sc)

  runApp(sc, sqlSC, daemonConfig, params['numPartitions'])

  sc.stop()
