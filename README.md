[![CI](https://github.com/Jaramilloh/DOD-ci-cd/actions/workflows/cicd.yml/badge.svg?branch=main)](https://github.com/Jaramilloh/DOD-ci-cd/actions/workflows/cicd.yml)

# DOD-ci-cd Development Container

Continuous integration and continuous delivery of the optimized ablation-driven depth object detector

## Description

This development container is designed for the DOD-ci-cd project. It provides a consistent, reproducible development environment for all team members.

For details in the architecture of the Optimized Depth Object Detector trained for fruits, please check [Jaramilloh/Depth-Object-Detector-DOD](https://github.com/Jaramilloh/Depth-Object-Detector-DOD) and our paper [Application of Machine Vision Techniques in Low-Cost Devices to Improve Efficiency in Precision Farming](https://doi.org/10.3390/s24030937).

## Requirements

- Docker
- Visual Studio Code
- Visual Studio Code Extension: Remote - Containers

## Usage

1. Clone the repository to your local machine.
2. Open the project in Visual Studio Code.
3. When prompted to reopen the project in the container, select 'Reopen in Container'.
4. Wait for the container to build and start. This may take a few minutes the first time.
5. You're now ready to start developing!

## Container Features

- Use the following commands defined on the [Makefile](Makefile)
    - make format
    - make lint
    - make test

## License

This project is licensed under the (AGPL-3.0 license) license. For more details, please see the [GNU AFFERO GENERAL PUBLIC LICENSE](LICENSE) file.

## Contact

If you have any questions or suggestions, please contact jfjarher@upv.edu.es.
