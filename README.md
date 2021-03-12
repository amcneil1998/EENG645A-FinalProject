# Introduction
This project is an image segmentation net for use with City Scapes

# Requirements
This project requires docker and Tensorflow 2.3.0.  Raw Data should be stored in:

        /opt/data/FinalProject/CityScapes
Both gtCoarse and gtFine will be used and should be unzipped in the City Scapes directory. The script will
perform a one time parse of the data into TF record files and store them in:

    /opt/data/CityScapes

# Running
Assuming that you have your system set up to allow tensorflow GPU access in a docker container run with:

    docker-compose up --build