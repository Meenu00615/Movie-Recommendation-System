#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
movies=pd.read_csv("top10K-TMDB-movies.csv")


# In[85]:


movies.head()


# In[86]:


movies.describe()


# In[87]:


movies.info()


# In[88]:


movies.isnull().sum()   # 13 null values in the overview column


# In[89]:


movies.columns  # To check the columns


# In[90]:


#id #title #genre
movies=movies[['id', 'title', 'genre', 'overview']]


# In[ ]:





# In[91]:


movies['tags']=movies['overview']+movies['genre']
#dfmi.loc[:, ('one', 'second')]
#movies


# In[92]:


new_data=movies.drop(columns=['overview', 'genre'])


# In[93]:


new_data


# # Algorithm

# 1. Bag of word
# 2. TFIDF

# In[94]:


from sklearn.feature_extraction.text import CountVectorizer


# In[95]:


cv=CountVectorizer(max_features=10000, stop_words='english')


# In[96]:


cv   # 10000 because of the 10000 movies


# In[97]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()


# In[98]:


vector.shape


# In[99]:


vector


# #for the similarity in the dataset

# In[100]:


from sklearn.metrics.pairwise import cosine_similarity   #theta


# In[101]:


similarity=cosine_similarity(vector)


# In[102]:


similarity   #similar value


# In[103]:


new_data[new_data['title']=='The Godfather'].index[0]
                                                    


# In[104]:


new_data[new_data['title']=='Captain America'].index[0]
                                                    


# In[105]:


#calculate the distance based on the similarity


# In[106]:


distance=sorted(list(enumerate(similarity[2])), reverse=True,key=lambda vector:vector[1])
for i in distance[0:5]:   #top five value
    print(new_data.iloc[i[0]].title) #with title


# In[107]:


def recommend(movies):
   # recommend
        index=new_data[new_data['title']==movies].index[0] #with title
        distance=sorted(list(enumerate(similarity[index])), reverse=True,key=lambda vector:vector[1])
        for i in distance[0:5]:   #top five value
            print(new_data.iloc[i[0]].title) #with title               


# In[108]:


recommend("Iron Man")


# In[109]:


#Now saving this file into the pickle file. So, that we can use it latter on for the program


# In[110]:


import pickle


# In[111]:


pickle.dump(new_data, open('movies_list.pkl','wb'))


# In[112]:


pickle.dump(new_data, open('similarity.pkl', 'wb'))


# In[113]:


pickle.load(open('movies_list.pkl', 'rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




