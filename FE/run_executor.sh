#!/bin/sh

# example commend for run app.
# --jars: explictly declear path to the mysql.jdbc driver (due to the SparkSQL
# --master: mode tu run app.
nohup spark-submit --jars /home/taey16/spark-1.6.1-bin-hadoop2.6/lib/mysql-connector-java-5.1.39-bin.jar --master spark://cluster_url:port executor_detector.py --driverMemory=6g --executorMemory=3g > log_texture_detector_800K.log &
