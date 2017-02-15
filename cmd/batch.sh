#!/bin/bash
set -x
LOG=/tmp/log.txt
rm $LOG
for i in $(seq 0 16);
do
echo ===================================== >>$LOG
echo $i >>$LOG
echo ===================================== >>$LOG
go test -run sin -tracecost -lessrnn $i &>>$LOG
done
