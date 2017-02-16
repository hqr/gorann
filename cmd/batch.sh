#!/bin/bash
set -x
###
# sinh-cosh
###
LOG=/tmp/log-sinh-cosh.txt
rm $LOG
for i in $(seq -1 4);
do
echo ===================================== >>$LOG
echo $i >>$LOG
echo ===================================== >>$LOG
go test -run sinh -lessrnn $i &>>$LOG
done
###
# sine-cosine
###
LOG=/tmp/log-sine-cosine.txt
rm $LOG
for i in $(seq -1 4);
do
echo ===================================== >>$LOG
echo $i >>$LOG
echo ===================================== >>$LOG
go test -run sine -lessrnn $i &>>$LOG
done
