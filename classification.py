import os
import numpy as np
import pandas as pd
directory_path=r"C:\Users\\abhin\Desktop\Projects\CNN_PYTORCH\train\\"
label_path=r"C:\Users\\abhin\Desktop\Projects\CNN_PYTORCH\labels.csv"
dataset=pd.read_csv(label_path)
try:
  os.mkdir(r"C:\Users\\abhin\Desktop\Projects\CNN_PYTORCH\dataset")
except:
  pass
for i in dataset["id"]:
  try:
    os.mkdir(r"C:\Users\\abhin\Desktop\Projects\CNN_PYTORCH\dataset\\"+dataset["label"][i-1])
  except:
    pass
  source=directory_path+str(i)+".png"
  destination=r"C:\Users\\abhin\Desktop\Projects\CNN_PYTORCH\dataset\\"+dataset["label"][i-1]+"\\"+str(i)+".png"
  os.rename(source,destination)