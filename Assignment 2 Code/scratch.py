import os

try:
  os.system('pause')  #windows, doesn't require enter
except whatever_it_is:
  os.system('read -p "Press any key to continue"') #linux
  
print("Function complete")
g