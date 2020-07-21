#!/bin/bash

base_path="build/logs/fit"
dirs=$(ls build/logs/fit)
echo $dir

for dir in $dirs; do
  echo $(ls -ls $base_path'/'$dir'/weights.last.hdf5')
  mkdir -p 'trained/logs/fit/'$dir
  cp $base_path'/'$dir'/weights.last.hdf5'  'trained/logs/fit/'$dir'/weights.last.hdf5'
  # for file in $(ls $base_path/$dir); do
  # 	echo $file
  # done
done

