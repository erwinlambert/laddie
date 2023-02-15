import sys
sys.path.append('src/')
from laddie import Laddie

#Check whether config file is specified
Nargs = len(sys.argv)
assert Nargs > 1, f"Need to specify config file"
assert Nargs < 3, f"Too many arguments"
assert sys.argv[1][-4:] == 'toml' , f"config file should be .toml"

laddie = Laddie(sys.argv[1])
laddie.compute()
