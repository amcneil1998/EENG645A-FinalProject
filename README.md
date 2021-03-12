# Introduction
This project is an image segmentation net for use with City Scapes

# Requirements
This project requires docker and Tensorflow 2.3.0.  Data should be stored in:

        /opt/FinalProject/CityScapes
Both gtCoarse and gtFine will be used and should be unzipped in the CityScapes directory.

# Running
Assuming that you have your system set up to allow tensorflow GPU access in a docker container run with:

    docker-compose up --build