#!/usr/bin/env python
# coding: utf-8

# # TikTok project
# 
# **Exploratory Data Analysis**

# **The purpose** of this project is to conduct exploratory data analysis on the provided data set and to continue the investigation with the aim of learning more about the variables. Of particular interest is information related to what distinguishes claim videos from opinion videos.
# 
# **The goal** is to explore the dataset and create visualizations.
# <br/>
# *This activity has 4 parts:*
# 
# **Part 1:** Imports, links, and loading
# 
# **Part 2:** Data Exploration
# *   Data cleaning
# 
# 
# **Part 3:** Build visualizations
# 
# **Part 4:** Evaluate and share results

# **Data exploration and cleaning**

# In[95]:


# Import packages for data maipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[96]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# Data exploration

# In[97]:


data.head()


# In[98]:


data.size


# In[99]:


data.shape


# In[100]:


data.info()


# In[101]:


data.describe()


# **Build visualizations**

# `video_duration_sec`

# In[102]:


# "video_duration_sec" boxplot
plt.figure(figsize=(5,1))
plt.title("video duration boxplot")
sns.boxplot(x=data["video_duration_sec"]);


# In[103]:


# "video_duration_sec" histogram
plt.figure(figsize=(5,3))
plt.title("video duration histogram")
sns.histplot(data["video_duration_sec"], bins=range(0,61,5));


# All videos are from 5 to 60 seconds duration and the distribution is uniform.

# `video_view_count`

# In[104]:


# "video_view_count" boxplot
plt.figure(figsize=(5,1))
plt.title("video view count boxplot")
sns.boxplot(x=data["video_view_count"]);


# In[105]:


# "video_view_count" histogram
plt.figure(figsize=(5,3))
plt.title("video view count histogram")
sns.histplot(data["video_view_count"], bins=range(0,(10**6+1),10**5));


# Distribution is right skewed.
# 
# More than 10,000 videos (a bit more than half the videos) receive fewer than 100,000 views.

# `video_like_count`

# In[106]:


# "video_like_count" boxplot
plt.figure(figsize=(10,1))
plt.title("video like count boxplot")
sns.boxplot(x=data["video_like_count"]);


# In[107]:


# "video_like_count" histogram
plt.figure(figsize=(5,3))
plt.title("video like count histogram")
ax = sns.histplot(data["video_like_count"], bins=range(0,(7*10**5+1),10**5))
labels = [0] + [str(i) + "k" for i in range(100,701, 100)]
ax.set_xticks(range(0,7*10**5+1,10**5), labels=labels);


# Distribution is right skewed.
# 
# Far more videos with less than 100K likes.

# Examine `video_comment_count`

# In[108]:


# "video_comment_count" boxplot
plt.figure(figsize=(5,1))
plt.title("video comment count boxplot")
sns.boxplot(x=data["video_comment_count"]);


# In[109]:


# "video_comment_count" histogram
plt.figure(figsize=(5,3))
plt.title("video comment count histogram")
sns.histplot(data["video_comment_count"], bins=range(0,(3001),100));


# Most of the videos have less than 100 comments, with the distribution again beign right-skewed.

# Examine `video_share_count`

# In[110]:


# "video_share_count" boxplot
plt.figure(figsize=(5,1))
plt.title("video share count boxplot")
sns.boxplot(x=data["video_share_count"]);


# In[111]:


# "video_share_count" histplot
plt.figure(figsize=(5,3))
plt.title("video share count histogram")
sns.histplot(data["video_share_count"], bins=range(0,(270001),10000));


# Most of the videos have 10,000 shares and less. The distribution is right-skewed.

# Examine `video_download_count`

# In[112]:


# "video_download_count" boxplot
plt.figure(figsize=(5,1))
plt.title("video download count")
sns.boxplot(x=data["video_download_count"]);


# In[113]:


# "video_download_count" histogram
plt.figure(figsize=(5,3))
plt.title("video download count histogram")
sns.histplot(data["video_download_count"], bins=range(0,(15001),500));


# The majority of the videos were downloaded fewer than 500 times. Data is skewed to the right.

# Examine `claim_status` by `verification_status`

# In[114]:


# `claim_status` by `verification_status` histogram
plt.figure(figsize=(7,4))
plt.title("claims by verification status histogram")
sns.histplot(x=data["claim_status"],
             hue=data["verified_status"],
             multiple="dodge",
             shrink=0.9);


# There are far fewer verified users than unverified users.
# 
# Verified users seem to be more likely to post opinions.

# Examine `claim_status` by `author_ban_status`

# In[115]:


# "claim_status" by "author_ban_status" histogram
plt.figure(figsize=(7,4))
plt.title("claims by author status histogram")
sns.histplot(x=data["claim_status"],
             hue=data["author_ban_status"],
             multiple="dodge",
             hue_order=["active", "under review", "banned"],
             palette={"active":"green", "under review":"orange", "banned":"red"},
             shrink=0.8,
             alpha=0.5);


# For both claims and opinions the most videos come from active users, with the opinion videos to be a bit more than claim videos.
# 
# For under review and banned authors, even though they don't have as many videos as active authors, the claim videos are much more than the opinion videos. It seems that authors that post claim videos are more likely to come under review or be banned.

# Examine `median view counts` by `author_ban_status`

# In[116]:


# calculate median
ban_status_counts = data.groupby(["author_ban_status"]).median(numeric_only=True).reset_index()

fig = plt.figure(figsize=(5,3))
plt.title("median view count by author ban status")
sns.barplot(data=ban_status_counts,
            x="author_ban_status",
            y="video_view_count",
            order=["active", "under review", "banned"],
            palette={"active":"green", "under review":"orange", "banned":"red"},
            alpha=0.5);


# The median view count for under review or banned authors is many times greater than that of active authors. Since we know that non-active authors are most likely to post claims, and that videos from non-active authors get far more views we can conclude that video view count may be a good indicator of claim status.

# Examine `video_view_count` by `claim_status`

# In[117]:


# check median "video_view_count" by "claim_status"
data.groupby("claim_status")["video_view_count"].median()


# In[119]:


# total views by "claim_status" piegraph
fig = plt.figure(figsize=(4,4))
plt.title("Total views by video claim status")
plt.pie(data.groupby("claim_status")["video_view_count"].sum(), labels=["claim", "opinion"]);


# Median view count for claim videos is 100x that of opinion videos.

# In[130]:


# "video_view_count" vs "video_like_count" by "claim_status" scatterplot
sns.scatterplot(x=data["video_view_count"],
                y=data["video_like_count"],
                hue=data["claim_status"],
                s=10,
                alpha=0.3);


# In[132]:


# "video_view_count" vs "video_like_count" for "claim_status" == opinion scatterplot
opinion = data[data["claim_status"]=="opinion"]

sns.scatterplot(x=opinion["video_view_count"],
                y=opinion["video_like_count"],
                s=10,
                alpha=0.3);


# **Determine outliers**
# 
# A common way to determine outliers is to calculate the IQR (interquartile range) and set a threshold that is 1.5 * IQR above the 3rd and below the 1st quartile.
# 
# In this TikTok dataset the values of the count variables are not normally distributed, they are skewed to the right. A way to deal with this is by calculating the median value for each variable and then add the 1.5 * IQR.

# In[133]:


# Calculate IQR, median, outlier threshold and number of outliers
count_cols = ["video_view_count",
              "video_like_count",
              "video_share_count",
              "video_download_count",
              "video_comment_count"]

for column in count_cols:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    median = data[column].median()
    outlier_threshold = median + (1.5*iqr)
    # count the number of values that exceed the outlier threshold
    outlier_count = (data[column] > outlier_threshold).sum()
    print("Number of outliers, {}: {}".format(column, outlier_count))
    #print(f'Number of outliers, {column}:', outlier_count)

