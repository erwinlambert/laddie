#!/bin/sh

# Experiment variables
export CURRENTDAY=0
export NDAYS=1
export BOOLREST='false'
export RESTFILE=dummy

if [[ "$OSTYPE" == "linux"* ]]; then
    export HOMEFOLDER='/usr/people/lambert/work/projects/laddie'
else
    export HOMEFOLDER='/Users/erwin/projects/laddie'
fi

# Loop over ice shelves
for GEOMYEAR in {0..5}
do

    export OLDFOLDER=$HOMEFOLDER'/output/isomip3/'
    export CONFIGTEMP=$HOMEFOLDER'/config_temp_'$GEOMYEAR'.toml'

    cp $HOMEFOLDER/config_isomip_tmpl.toml $CONFIGTEMP

    if [[ "$OSTYPE" == "linux"* ]]; then
        sed -i s/@GEOMYEAR/$GEOMYEAR/g $CONFIGTEMP
        sed -i s/@NDAYS/$NDAYS/g $CONFIGTEMP
        sed -i s/@BOOLREST/$BOOLREST/g $CONFIGTEMP
        sed -i s#@RESTFILE#$RESTFILE#g $CONFIGTEMP
    else
        sed -i '' s/@GEOMYEAR/$GEOMYEAR/g $CONFIGTEMP
        sed -i '' s/@NDAYS/$NDAYS/g $CONFIGTEMP
        sed -i '' s/@BOOLREST/$BOOLREST/g $CONFIGTEMP
        sed -i '' s#@RESTFILE#$RESTFILE#g $CONFIGTEMP
    fi

    echo Starting $GEOMYEAR $FORCING
    python3 runladdie.py $CONFIGTEMP

    rm $CONFIGTEMP

    echo Finished $GEOMYEAR $FORCING

    CURRENTDAY=$(($CURRENTDAY + $NDAYS))
    RESTDAY=$(printf "%06d" $CURRENTDAY)
    echo $CURRENTDAY
    echo $RESTDAY
    export BOOLREST='true'
    export RESTFILE=$OLDFOLDER'restart_'$RESTDAY'.nc'
    echo $RESTFILE

done