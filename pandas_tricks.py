### Count column values. For example column with different technologies

import collections
import itertools

sample_data = pd.read_csv("data/sample_data.csv")
#sample_data.columns = map(str.lower, sample_data.columns)

counter = collections.Counter(itertools.chain.from_iterable(v.split(',') for v in sample_data.techs))

##Now keys from counter object will be tech and values will be the frequency of the tech in the techs column

counter_list =[list(counter.keys()),list(counter.values())]

#save the final result in a file
pd.DataFrame(np.array(counter_list).T, columns=['Technology','Count']).sort_values(by='Count',ascending=False).to_csv('tech_list.csv',index=False)
