#!/bin/bash

for f in *
do
    new=`echo "${f}" | sed -e "s/$1/$2/g"`
    mv $f $new
done
