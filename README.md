## modulo_III

Package to deploy a model

Initially appeared on
[gist](https://github.com/RoSaav/modulo_III).

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- Python 3.9.x

### Installing - Microsoft Windows

A step by step series of examples that tell you how to get a development
environment running

Run the next lines in a powershell

-pip install virtualenv

-virtualenv todos_env

-todos_env\Scripts\activate

-pip install -r requirements.txt

## Structure

MODULO_III
-TITANIC
--app: Front-end deploymment with FastApi
--data Initial data set Titanic
--server Server for response deploymment with FastApi
--train Training a machine learning model
--utils Transformers for preprocessing
 
## Running the tests

Every child carpet of Titanic is build as a individual project, so if you want to test every project, you have to zoom in in the folder

### Sample Tests

-todos\tests\test_list.py .........            [ 47%]

-todos\tests\integration\test_integration.py ..[ 57%]

-todos\tests\unit\test_add_task.py ..          [ 68%]

-todos\tests\unit\test_create.py ......        [100%]

-========================================== 19 passed, 5 warnings in 0.98s -=========================================== 
-______________________________________________________ summary -______________________________________________________

-  py39: commands succeeded

-  congratulations :)

## Deployment

The file include a docker compose to deployd a infrastructure of rest Api 

## Authors

Homework for MODULO_III

TECNOLOGICO DE MONTERREY

Miguel Rodrigo Saavedra Perez

https://github.com/RoSaav/modulo_II


## License

This project is licensed under the [MIT](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details
