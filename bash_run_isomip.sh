#!/bin/sh

# Experiment variables
export RUNNAME='isomip3' #Run name
export NDAYS=1 #Number of days to run for each year

# Variables that do not need to be changed
export CURRENTDAY=0 #Track day of simulation, start at 0
export BOOLREST='false' #Do not use restart file during first run. Set to 'true' after first run
export RESTFILE=dummy #Some dummy value, changed after first run.

#Define proper home folder
if [[ "$OSTYPE" == "linux"* ]]; then
    export HOMEFOLDER='/usr/people/lambert/work/projects/laddie'
else
    export HOMEFOLDER='/Users/erwin/projects/laddie'
fi

#Folder in which output is saved
export OLDFOLDER=$HOMEFOLDER'/output/'$RUNNAME'/'

# Loop over ice shelves
for GEOMYEAR in {0..2}
do

    #Temporary configuration file
    export CONFIGTEMP=$HOMEFOLDER'/config_temp_'$GEOMYEAR'.toml'

    #Copy template
    cp $HOMEFOLDER/config_isomip_tmpl.toml $CONFIGTEMP

    #Overwrite run-specific parameters
    if [[ "$OSTYPE" == "linux"* ]]; then
        sed -i s/@GEOMYEAR/$GEOMYEAR/g $CONFIGTEMP
        sed -i s/@NDAYS/$NDAYS/g $CONFIGTEMP
        sed -i s/@BOOLREST/$BOOLREST/g $CONFIGTEMP
        sed -i s#@RESTFILE#$RESTFILE#g $CONFIGTEMP
        sed -i s#@RUNNAME#$RUNNAME#g $CONFIGTEMP
    else
        sed -i '' s/@GEOMYEAR/$GEOMYEAR/g $CONFIGTEMP
        sed -i '' s/@NDAYS/$NDAYS/g $CONFIGTEMP
        sed -i '' s/@BOOLREST/$BOOLREST/g $CONFIGTEMP
        sed -i '' s#@RESTFILE#$RESTFILE#g $CONFIGTEMP
        sed -i '' s#@RUNNAME#$RUNNAME#g $CONFIGTEMP
    fi

    #Run Laddie
    echo Starting $GEOMYEAR
    python3 runladdie.py $CONFIGTEMP

    #Remove temporary configuration file
    rm $CONFIGTEMP

    echo Finished $GEOMYEAR

    #Compute restart day to call correct restart file
    CURRENTDAY=$(($CURRENTDAY + $NDAYS))
    RESTDAY=$(printf "%06d" $CURRENTDAY)
    export BOOLREST='true'
    export RESTFILE=$OLDFOLDER'restart_'$RESTDAY'.nc'

done