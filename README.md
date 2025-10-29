# Xenobot Movement Analysis Using Transition Probability Matrices

Refurbished pipeline for analyzing changes in xenobot movement after applying stimulus.

## Description

Taking time-lapse videos of xenobot movements over long periods of time, their movement
coordinates are tracked and recorded. Each video is divided up into 30-second chunks and
analyzed, checking for changes in movement behavior.

## Getting Started

### Dependencies
# Make sure to download Python3.12 on your device
For MacOS, it's the following command (assuming you have HomeBrew).
```
brew install Python3.12
```

# Set Up a Local Python Environment 
```
cd movement_visualizer
```
```
python3.12 -m venv .venv
```
Activate the Python environment.
```
source .venv/bin/activate
```
Download the required dependencies to run these files.
```
pip install -r requirements.txt
```

### Executing program

How to run the program:
* use trackR to obtain movement metrics/coordinates of xenobots
* use trackFixer to revise xenobot tracking annotations
* move into the correct directory
```
cd movement_visualizer
```
* run the calculate.py file to obtain linear speed, headings, & angular speed 
```
python calculate.py
```
* using the calculated metrics, run graph.py to visualize the changes in states
```
python graph.py
```

## Help

Please contact me at yu.zeng@tufts.edu for any issues or concerns.

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)