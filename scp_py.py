import os
import glob
import sys

path_list = glob.glob(f'{sys.argv[1]}')
for path in path_list:
    #  print(path)
    #  os.system(f"scp -i /home/yryrgogo/gixo-hagihara.pem {path} ubuntu@13.231.227.78':/home/ubuntu/")
    os.system(f"scp -i /home/yryrgogo/gixo-hagihara.pem {path} ubuntu@13.231.56.86:/home/ubuntu/{sys.argv[2]}")
