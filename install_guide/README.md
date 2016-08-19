# Install java
- Ref: http://askubuntu.com/questions/521145/how-to-install-oracle-java-on-ubuntu-14-04
# Install Scala
- Hadoop precompiled binary로 Spark를 설치할 경우, 해당 바이너리를 빌드하기위해 사용된 scala의 버전을 확인하여 맞춰줘야 함.
	- **spark1.6.1 Prebuild for Hadoop2.6 에서는 scala-2.11.x를 설치하면 안됨!!**
- Ref: https://gist.github.com/osipov/c2a34884a647c29765ed
# Istall Spark
- Download: http://spark.apache.org/downloads.html
- version: spark1.6.1 Prebuild for Hadoop2.6
- bash_profile 수정필요 (62/63서버의 bash_profile 참조)

# Spark cluster setup:
- Ref: http://spark.apache.org/docs/latest/spark-standalone.html#cluster-launch-scripts
- `spark-env.sh` 수정
- `slave` 파일 작성
- `log4j~` 파일 수정
- spark cluster mode start: (webUI: master_server_id:8080)
    - /path/to/spark_home/sbin/start-all.sh

# Hadoop install
- (spark는 기본으로 hadoop 바이너리를 포함하지만 hdfs를 쓰기위해 설치했습니다.)
- hadoop2.6.0으로 설치 및 실행 (WebUI: http://10.202.34.62:50070)
- Ref(single node cluster): http://thepowerofdata.io/setting-up-a-apache-hadoop-2-7-single-node-on-ubuntu-14-04/
- Ref(Multi node cluster): http://hadooptutorial.info/install-hadoop-on-multi-node-cluster/
