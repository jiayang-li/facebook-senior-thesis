#!/usr/bin/env python
# coding: utf-8

# # Pre-Processing Data

# ## Import Data
# This data was collected through "Survey 1 - Rating Facebook Content Appropriateness"

# In[35]:


# General Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

# SK Learn Libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Other Libraries
from astropy.table import Table
from astropy.io import ascii
from collections import Counter
from collections import OrderedDict


# In[36]:


# Create Bag of Words Numpy Array
df = pd.read_csv("Data/results.csv", delimiter=",")


# ## Edit Data

# ### Drop rows that do not contain information about question responses.

# Drop the first 151 rows of responses due to survey response issues that would violate research integrity.

# In[37]:


# Drop Rows
drop_rows = list(range(0, 151))
df.drop(drop_rows, inplace=True)
df.reset_index(drop=True, inplace=True)


# ### Drop rows for responses that took less than 400 seconds to complete

# In[38]:


drop_400 = []
for i in range(2, len(df.index)):
    if int(df.iloc[i, 5]) < 400: 
        drop_400.append(i)
df.drop(drop_400, inplace=True)


# In[39]:


df


# ### Restructure data

# In[40]:


# Export edited data 
df.to_csv('Data/results_edited_dropped_400.csv', index=False)


# In[41]:


# Drop Columns
df = pd.read_csv("Data/results_edited_dropped_400.csv", delimiter=",")
drop_columns = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage', 'Q39', 'Q40', 'Q40 - Topics', 'Q41']
df = df.drop(columns=drop_columns, axis=1)
df.shape


# ### Replace remaining NaNs

# In[42]:


# Replace NaN's with the mode
# df = df.fillna(df.mode().iloc[0])
df.dropna(inplace=True)
df


# In[43]:


# Export edited data 
df.to_csv('Data/results_no_metadata_dropped_400.csv', index=False)


# In[44]:


# Convert all data to integers
df = pd.read_csv("Data/results_no_metadata_dropped_400.csv", delimiter=",", dtype=int)
response_numbers = len(df)
df = df.transpose()
index = df.index.values
df.shape


# ### Isolate Community Standards

# In[45]:


def isolate_data(a, b):
    data = df.iloc[a:b]
    return data, data.index.values


# In[46]:


hate, hate_index = isolate_data(0, 5)
nudity, nudity_index = isolate_data(5, 10)
cruel, cruel_index = isolate_data(10, 15)
spam, spam_index = isolate_data(15, 20)
false, false_index = isolate_data(20, 25)


# ## Analyze Data

# ### Means and Variances

# In[47]:


# Define function for getting means and variances
def get_stat(data):
    means = []
    sds = []    
    for i in range(len(data)):
        m = np.round(np.mean(data.iloc[i]), decimals=4)
        sd = np.round(np.std(data.iloc[i]), decimals=4)
        means.append(m)
        sds.append(sd) 
    return means, sds


# In[48]:


# Get means and standard deviations for overall dataset
means, sds = get_stat(df)   

# Get means and standard deviations for each Community Standard
hate_means, hate_sds = get_stat(hate)
nudity_means, nudity_sds = get_stat(nudity)
cruel_means, cruel_sds = get_stat(cruel)
spam_means, spam_sds = get_stat(spam)
false_means, false_sds = get_stat(false)

#for i in range(len(index)):
    #print('%s: means=%.2f sd=%.2f' %(index[i], means[i], sds[i]))


# ### Figures for Means and Variances

# In[49]:


# Define function to create bar graph with error bars   
def plot_figs(index, means, sds, standard, file_name):
    fig = plt.figure(figsize=(10,5))
    plt.bar(index, means, yerr=sds, align='center', alpha=0.8, ecolor='black', capsize=10, color='tab:purple')
    plt.ylabel('Rating of Facebook Posts')
    plt.xticks(index)
    plt.title('Means and Standard Devations for %s (Response Time > 400 sec)' % standard)
    plt.tight_layout()
    plt.savefig('Figures_dropped_400/%s_bar_plot_with_error_bars_dropped_400.png' % file_name, overwrite=True)
    plt.show()

# Plot and save figures
plot_figs(index, means, sds, 'All Five Standards', 'all')  
plot_figs(hate_index, hate_means, hate_sds, 'Hate Speech', 'hate')
plot_figs(nudity_index, nudity_means, nudity_sds, 'Adulty Nudity and Sexual Activity', 'nudity')
plot_figs(cruel_index, cruel_means, cruel_sds, 'Cruel and Insensitive Content', 'cruel')
plot_figs(spam_index, spam_means, spam_sds, 'Spam', 'spam')
plot_figs(false_index, false_means, false_sds, 'False News', 'false')


# In[50]:


def create_stat_charts(index, means, sds, standard, file_name):
    t = Table([index, means, sds], names=('index', 'means', 'sds'), meta={'name': 'Table on Survey 1 %s Means and Variances (Response Time > 400 seconds)' % file_name})
    ascii.write(t, 'Tables_dropped_400/table_of_%s_means_sds_dropped_400.csv' % file_name, format='csv', fast_writer=False, overwrite=True)
    print(t)
# Create charts with means and variances
create_stat_charts(index, means, sds, 'All', 'all')
create_stat_charts(hate_index, hate_means, hate_sds, 'Hate Speech', 'hate')
create_stat_charts(nudity_index, nudity_means, nudity_sds, 'Adult Nudity and Sexual Activity', 'nudity')
create_stat_charts(cruel_index, cruel_means, cruel_sds, 'Cruel and Insensitive Speech', 'cruel')
create_stat_charts(spam_index, spam_means, spam_sds, 'Spam', 'spam')
create_stat_charts(false_index, false_means, false_sds, 'False News', 'false')


# ### Bar Graphs for Each Question

# In[51]:


def get_counts(data):
    prev = 0
    final_counts = []
    counts = Counter(data.iloc[0])
    counts = OrderedDict(sorted(counts.items()))
    for key, value in counts.items():
        if key-1 != prev:
            final_counts.append(0)
        prev = key
        final_counts.append(value)
    return final_counts

def graph(data):
    for i in range(25):
        data, index = isolate_data(i, i+1)
        counts = get_counts(data)
        bar_spacing = list(range(1, 8, 1))
        plt.bar(bar_spacing, counts, width=0.75, color='tab:purple')
        for j in range(7):
            plt.text(x=bar_spacing[j]-0.2, y=counts[j]+6, s=counts[j], fontsize=12)
        plt.xlabel('Rankings', fontsize=14)
        plt.ylabel('Frequencies', fontsize=14)
        plt.title('Rankings for Survey %s (Response Time > 400 Seconds)' % index[0], fontsize=14)
        plt.text(3, 330,'Number of Responses = %i' % response_numbers, fontsize=14)
        plt.text(5, 300,'mean = %.4f' % means[i], fontsize=14)
        plt.text(5, 270,'sd = %.4f' % sds[i], fontsize=14)
        plt.axis([0, 8, 0, 350])
        plt.grid(True, ls="-", color='0.8')
        plt.savefig('Figures_dropped_400/%s_ranking_bar_plot_dropped_400.png' % index[0], overwrite=True)
        plt.show()
        
graph(df)


# In[ ]:





# In[ ]:





# In[ ]:




