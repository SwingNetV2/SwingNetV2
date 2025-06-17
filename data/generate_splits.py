from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split


x = loadmat('golfDB.mat') 
l = list(x['golfDB'][0])
d = dict()
for idx, k in enumerate(l):
    d["{:3d}".format(idx)] = list(l[idx])
df = pd.DataFrame(d).T
df.columns = ["id","youtube_id","player", "sex", "club","view","slow","events","bbox","split"]

df['id'] = df['id'].apply(lambda x: x[0][0])
df['youtube_id'] = df['youtube_id'].apply(lambda x: x[0])
df['player'] = df['player'].apply(lambda x: x[0])
df['sex'] = df['sex'].apply(lambda x: x[0])
df['club'] = df['club'].apply(lambda x: x[0])
df['view'] = df['view'].apply(lambda x: x[0])
df['slow'] = df['slow'].apply(lambda x: x[0][0])
df['events'] = df['events'].apply(lambda x: x[0])
df['bbox'] = df['bbox'].apply(lambda x: x[0])
df['split'] = df['split'].apply(lambda x: x[0][0])


df.index = df.index.astype(int)
df.to_pickle('golfDB.pkl')

df = pd.read_pickle('golfDB.pkl')
train_split, val_split = train_test_split(df, test_size=0.2, random_state=42)
train_split = train_split.reset_index()
train_split = train_split.drop(columns=['index'])
val_split = val_split.reset_index()
val_split = val_split.drop(columns=['index'])
train_split.to_pickle('train_split.pkl')
val_split.to_pickle('val_split.pkl')
