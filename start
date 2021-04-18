#!/bin/bash

git clone https://github.com/riveSunder/carle.git

cd carle
pip install .

cd ../

python -c "from carle.env import CARLE; env = CARLE(); print('Looks OK!')"

exec "$@"
