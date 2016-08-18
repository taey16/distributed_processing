#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Do not use this code!!!
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

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/602.1.25 (KHTML, like Gecko) Version/9.1.1 Safari/601.6.10'}

requestPrefix = \
  'http://%s:%d/request_handler?url=http://i.011st.com/%s&attribute=color_signature&%s'
daemonConfig = {'host': ['url', 'url'],
                'port': [8081, 7081, 6081, 5081]}

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


def process(info):
  reload(sys)
  sys.setdefaultencoding('utf-8')

  global daemonConfig
  global headers

  port = daemonConfig['port'][random.randint(0, len(daemonConfig['port'])-1)]
  host = daemonConfig['host'][random.randint(0, len(daemonConfig['host'])-1)]

  try:
    apiParams = urllib.urlencode({'lctgr_nm': info[1], 
                                  'mctgr_nm': info[2], 
                                  'sctgr_nm': info[3]})
    requestURL = requestPrefix % (host, port, info[0], apiParams)
    print(requestURL)

    opener = urllib2.build_opener()
    opener.addheaders = headers.items()
    response = opener.open(requestURL) 
    retrievedItems = json.loads(response.read())

    successFlag = False
    successFlag = sanityCheck(retrievedItems, requestURL)
    if successFlag:
      #return retrievedItems
      return json.dumps(retrievedItems)
      #return retrievedItems
  except Exception as err:
    print('ERROR: %s' % err)


def dumpJSONForEachPartition(jsonDataIterator):
  with open('temp.json', 'a+') as fp:
    for jsonData in jsonDataIterator:
      fp.write(unicode(json.dumps(jsonData)) + unicode('\n'))

def dumpJSONForEach(jsonData):
  return unicode(json.dumps(jsonData)) + unicode('\n')


def runApp(sc, sqlSC, daemonConfig):
  df = sqlSC.read\
            .format('jdbc')\
            .options(url='jdbc:mysql://url:3306/DBName?user=oops&password=oops',
                     dbtable=u'(select __org_img_url__, __lctgr_nm__, __mctgr_nm__, __sctgr_nm__ from 11st_fashion_tab where __lctgr_nm__ = \'xxx\' limit 180) df',
                     driver='com.mysql.jdbc.Driver')\
            .load()\
            .collect()

  inputRDD = sc.parallelize(df, 18).cache()
  #rdd = sc.parallelize(df).map(lambda x: url_prefix(x[0]))
  #rdd = sc.parallelize(df).map(url_prefix).collect()
  print 'Finished parallelize %d' % inputRDD.count()

  outputRDD = inputRDD.map(process)
  #outputRDD.foreach(dumpJSONForEach)
  #outputRDD.saveAsTextFile('hdfs://url:9000/tmp.json')
  #sqlSC.jsonRDD(outputRDD).write.save('hdfs://url:9000/tmp/tmp.json')
  print 'Finished'


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--executorMemory', 
                      default='6G', help='executor-memory')
  parser.add_argument('--driverMemory', 
                      default='3G', help='driver-memory')
  args = parser.parse_args()
  params = vars(args)

  conf = SparkConf() \
    .setAppName("exam_mysql")\
    .set("spark.executor.memory", params['executorMemory']) \
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)
  sqlSC = SQLContext(sc)

  runApp(sc, sqlSC, daemonConfig)

  sc.stop()
