# Introduction
This project is an image segmentation net for use with City Scapes

# Requirements
This project requires docker and Tensorflow 2.3.0.  Raw Data should be stored in:

        /opt/data/FinalProject/CityScapes
Both gtCoarse, gtFine, left8BitImg will be used and should be unzipped in the City Scapes directory. The script will
perform a one time parse of the data into TF record files and store them in:

    /opt/data/CityScapes

These directories should exist before running.  This project also requires lots of disk space.  With retaining the original zip
folders in the /opt/data/FinalProject/CityScapes folder this folder is now 122.3GB.
Creating the TF record files takes around 321GB of disk space in /opt/data/CityScapes.  Make sure
that this space is available before starting to run the script.  Running the script takes a while
to make the TF record files.  On my machine it took around an hour.
# Running
Assuming that you have your system set up to allow tensorflow GPU access in a docker container run with:

    docker-compose up --build

# The mount commands
I have my data spread acros several volumes

    sudo mount /dev/nvme0n1p4 /opt/data/CityScapes/
    sudo mount /dev/sda2 /opt/data/FinalProject/