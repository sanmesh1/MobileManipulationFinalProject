# Mobile Manipulation Project

I basically combined the atsushi astar code with our project 4 robot code.Not done combining.

i think this onlyworks in linux environment.

## Installation

create a virtual environment through "python3.7 -m venv venv" inside this github folder.It will create a virtual environment with python 3.7. 

activate the environment through "source venv/bin/activate"

install all dependencies through "pip3.7 install -r requirements.txt". Im pretty sure this contains more packages than needed, so need to figure out how to avoid this.

If installing dependencies this way doesnt work, need to do it manually below:

pip3.7 install --upgrade pip
pip3.7 install https://files.pythonhosted.org/packages/e6/30/9c053e755e659e5bf5b7276c23b10bbc8e284ab8b85039e7d8e102d8517b/gtsam-4.0.2-cp37-none-manylinux2014_x86_64.whl
pip3.7 install matplotlib
pip3.7 install IPython
pip3.7 install sympy
pip3.7 install scipy


## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
