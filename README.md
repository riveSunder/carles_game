# Carle's Game

```
<(o)__
 ( .>/
```
## Installation

I am testing with and recommend using `virtualenv` to create python virtual environmeheight=128, width=128)nts for managing dependencies. If you already have `virtualenv` installed, make and activate a new environment like this:

```
virtualenv carles_venv --python=python3
source carles_venv/bin/activate 
```

Note that I am using Ubuntu 18. If you have a different experience setting up or using CARLE and Carle's Game, feel free to send me your notes (go ahead and open an issue) and I will update the installation instructions here. 

```
git clone  https://github.com/riveSunder/carles_game.git
cd carles_game
pip install -r requirements.txt

# install the environment, CARLE
git clone https://github.com/riveSunder/carle.git

cd carle

pip install -e .

# run tests
python -m test.test_all

# go back to the root directory
cd ../
python -c "from carle.env import CARLE; env = CARLE(); obs = env.reset(); print('Looks OK')"
```


##

The current evaluation template is a Jupyter notebook using Bokeh for interactive plotting. To launch a Jupyter notebook session:

```
jupyter notebook
```

However the scheme for running on notebook server [mybinder.org](https://mybinder.org) is a little more involved and uses `bokeh serve`. You can create an interactive bokeh app server on mybinder by following the link below.

[`https://mybinder.org/v2/gh/riveSunder/carles_game/requirements-0x01?urlpath=/proxy/5006/carles-game`](https://mybinder.org/v2/gh/riveSunder/carles_game/requirements-0x01?urlpath=/proxy/5006/carles-game)
