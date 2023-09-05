import pandas as pd
import numpy as np
data = pd.read_csv(r'D:\cuhk\23Spring\DDA4210\Project\DMSC.csv')
import re
def filter_emoji(desstr,restr=''):  
    #过滤表情   
    try:  
        co = re.compile(u'[\U00010000-\U0010ffff]|\u200b')  
    except re.error:  
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')  
    return co.sub(restr, desstr)


def clean(x):
    return filter_emoji(x,restr='')

data = data.astype(str)

data = data.apply(np.vectorize(clean))
data.to_csv(r'D:\cuhk\23Spring\DDA4210\Project\data_no_emoji.csv', index = False)


