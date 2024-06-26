#!/usr/bin/env python
# coding: utf-8

# # TikTok project
# 
# **Exploratory Data Analysis**

# **The purpose** of this project is to investigate and understand the data provided.
# 
# 1.   Get acquainted with the data
# 
# 2.   Compile summary information about the data
# 
# 3.   Begin the process of EDA and reveal insights contained in the data
# 
# 4.   Prepare for more in-depth EDA, hypothesis testing, and statistical analysis
# 
# **The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided TikTok information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 

# **Identify data types and compile summary information**

# In[20]:


# Import packages
import pandas as pd
import numpy as np


# In[21]:


data = pd.read_csv("tiktok_dataset.csv")


# In[22]:


data.head(10)


# In[23]:


data.info()


# In[24]:


data.describe()


# **Investigate the variables**
# 
# The ultimate objective of the project is to use machine learning to classify videos as either claims or opinions. A good place to begin with is determining how many videos there are of each different claim status.

# In[25]:


# Check values for `claim_status`
data["claim_status"].value_counts()


# Two values, "claim" and "opinion" that are quite balanced.
# 
# Next, examine the engagement trends associated with each different claim status.

# In[26]:


# Average view count of videos with "claim" status

claims = data[data["claim_status"] == "claim"]
print("Claims mean view count:", claims["video_view_count"].mean())
print("Claims median view count:", claims["video_view_count"].median())


# In[27]:


# Average view count of videos with "opinion" status

opinions = data[data["claim_status"] == "opinion"]
print("Opinions mean view count:", opinions["video_view_count"].mean())
print("Opinions median view count:", opinions["video_view_count"].median())


# Mean and median view count for the same `"claim_status"` are close to one another, but `claims` have much more views than `opinions`

# Examine trends associateed with the ban status of the author.

# In[28]:


# Get counts for every group combination of "claim_status" and "author_ban_status"

data.groupby(["claim_status", "author_ban_status"]).count()[["#"]]


# There are many more `claim` videos with `banned` `"author_ban_status"` than there are `opinion` videos of the same status.
# 
# We cannot be sure of why is that.
# 
# It is possible that `claim` videos are more strictly policed than `opinion` videos, or authors that post a `claim` video are under a stricter set of rules than if they posted an `opinion` video.
# 
# Also, we cannot know for sure if `claim` videos are more prone to be banned or if authors that post `claim` videos are most likely to violate terms of service and be banned.

# Investigate engagement levels by `"author_ban_status"`.

# In[47]:


# Calculate the mean, and the median video view, like, and share by "author_ban_status"

data.groupby(["author_ban_status"]).agg({"video_view_count": ["count", "mean", "median"],
                                         "video_like_count": ["count", "mean", "median"],
                                         "video_share_count": ["count", "mean", "median"]})


# In[42]:


# Median video_share_count by author_ban_status

data.groupby(["author_ban_status"]).median(numeric_only=True)[["video_share_count"]]


# The median share count of `banned` authors is 33 times the median share count of `active` authors.

# `banned` and `under review` authors get far more views, likes, and shares than `active` authors.
# 
# Also, in most groups the mean is much greater than the median which indicates that there are videos with very high engagement counts.

# Examine engagement rates.

# In[45]:


# Create likes_per_view column
data["likes_per_view"] = data["video_like_count"] / data["video_view_count"]

# Create comments_per_view column
data["comments_per_view"] = data["video_comment_count"] / data["video_view_count"]

# Create shares_per_view column
data["shares_per_view"] = data["video_share_count"] / data["video_view_count"]


# In[48]:


# Calculate mean and median of new engagement columns by "claim" and "ban" status

data.groupby(["claim_status", "author_ban_status"]).agg({"likes_per_view": ["count", "mean", "median"],
                                                         "comments_per_view": ["count", "mean", "median"],
                                                         "shares_per_view": ["count", "mean", "median"]})


# We know that videos by banned authors and those under review tend to get far more views, likes, and shares than videos by non-banned authors. However, *when a video does get viewed*, its engagement rate is less related to author ban status and more related to its claim status.
# 
# Also, we know that claim videos have a higher view rate than opinion videos, but this tells us that claim videos also have a higher rate of likes on average, so they are more favorably received as well. Furthermore, they receive more engagement via comments and shares than opinion videos.
# 
# Note that for claim videos, banned authors have slightly higher likes/view and shares/view rates than active authors or those under review. However, for opinion videos, active authors and those under review both get higher engagement rates than banned authors in all categories.

# * Of the 19,382 samples in this dataset, just under 50% are claims&mdash;9,608 of them.  
# * Engagement level is strongly correlated with claim status. This should be a focus of further inquiry.
# * Videos with banned authors have significantly higher engagement than videos with active authors. Videos with authors under review fall between these two categories in terms of engagement levels.
