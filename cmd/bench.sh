#!/bin/bash
go test -run hart -int1 1 2 >/dev/null
for i in `seq 5 5 30`; do go test -run hart -int1 $i 2 >/dev/null; done
