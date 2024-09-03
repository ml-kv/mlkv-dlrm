#!/bin/sh

for i in $(seq 0 1 23)
do
echo "Begin DownLoad Criteo Terabyte Data Day ${i}"
wget "https://sacriteopcail01.z16.web.core.windows.net/day_${i}.gz"
echo "Begin Unzip Criteo Terabyte Data Day ${i}"
gzip -d "day_${i}.gz"
echo "Get Criteo Terabyte Data Day ${i} Success"
done
