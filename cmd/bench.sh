#!/bin/bash
set -v
# 20x20
go test -run rcl -timeout 6h -int1 20 &>/tmp/20-20.log
# 20x20 plus random selection
go test -run rcl -timeout 6h -int1 20 -int2 1 &>/tmp/20-20-R.log
# 30x30
go test -run rcl -timeout 6h -int1 30 &>/tmp/30-30.log
# 30x30 plus random
go test -run rcl -timeout 6h -int1 30 -int2 1 &>/tmp/30-30-R.log
# 60x60
go test -run rcl -timeout 6h -int1 60 &>/tmp/60-60.log
# 60x60 plus random
go test -run rcl -timeout 6h -int1 60 -int2 1 &>/tmp/60-60-R.log
