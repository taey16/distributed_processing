#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
  'http://%s:%d/imgDetect?url=http://%s&is_browser=0'

imageURLPrefix = 'oops%s'
daemonConfig = {'host': ['10.xxx.xx.xx'],
                'port': [12345, 12346, 12347, 12348]}


def sanityCheck(item, requestURL):
  successFlag = True

  if item['result_roi'] == False:
    print('ERROR: in result[result_roi], req: %s' % requestURL)
    successFlag = False
  elif item['roi'] == {} or item['roi'] == None:
    print('ERROR: in result[roi], req: %s' % requestURL)
    successFlag = False

  return successFlag

#def roiParse(jsonFile):
#  targetTable = '11st_fashion_tab_texture'
#
#  jsonDict= json.loads(jsonFile)
#  roiDict = jsonDict['roi']
#  url = jsonDict['url']
#  key = url.split('PBrain')[1]
#
#  for rois in roiDict:
#    for roi in rois:
#      roiText = ''
#      for i in range(0, 4):
#        roiText = roiText + str(int(roi[i])) + ' '
#      roiText = roiText[:-1]
#
#      # for first roi, update the row already exists
#      if roi == roiDict[roiDict.keys()[0]][0]:
#        #update
#        sqlSC.sql('UPDATE %s SET __roi__="%s" WHERE __org_img_url__=%s' % (targetTable, roiText, url))
#      else:
#        sqlSC.sql('INSERT INTO %s (__prd_no__, __org_img_url__, __roi__,  __lctgr_nm__, __mctgr_nm__, __sctgr_nm__) SELECT __prd_no__, __org_img_url__, "%s" __lctgr_nm__, __mctgr_nm__, __sctgr_nm__ FROM %s WHERE __org_img_url__=%s' % (targetTable, roiText, targetTable, url))
#        # copy & update
#  return jsonDict['roi']

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
    imageURL = imageURLPrefix % entry[0]
    requestURL = requestPrefix % (host, port, imageURL)
    print(requestURL)

    opener = urllib2.build_opener()
    opener.addheaders = headers.items()
    response = opener.open(requestURL) 
    retrievedItems = json.loads(response.read())

    successFlag = sanityCheck(retrievedItems, requestURL)
    if successFlag:
      outputJsonData = {'url': retrievedItems['url'], 'roi': retrievedItems['roi']}
      #return json.dumps(retrievedItems, separators=(',', ':'))
      return json.dumps(outputJsonData, separators=(',', ':'))
  except Exception as err:
    print('ERROR: %s' % err)


def runApp(sc, sqlSC, daemonConfig, numPartitions=8):
  inputDF = sqlSC.read\
            .format('jdbc')\
            .options(url='jdbc:mysql://url?user=oops&password=oops',
                     dbtable=u'(select __org_img_url__ from 11st_fashion_tab_texture2 limit 800000) inputDF',
                     driver='com.mysql.jdbc.Driver')\
            .load()\
            .collect()

  inputRDD = sc.parallelize(inputDF, numPartitions).cache()
  totalSamples = inputRDD.count()
  print 'Finished parallelize %d' % totalSamples
  print '# of Partitions: %d' % inputRDD.getNumPartitions()

  outputRDD = inputRDD.map(process)\
                      .filter(lambda x: x <> None)
  outputRDD.saveAsTextFile('hdfs://url:9000/texture_detector')
  print 'Finished'


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
    .setAppName("executor_detect")\
    .set("spark.executor.memory", params['executorMemory']) \
    .set("spark.driver.memory", params['driverMemory'])
  sc = SparkContext(conf=conf)
  sqlSC = SQLContext(sc)

  runApp(sc, sqlSC, daemonConfig, params['numPartitions'])

  sc.stop()
