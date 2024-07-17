#!/usr/bin/env python
# coding: utf-8

# # TikTok project
# 
# **Data Exploration and Hypothesis Testing**

# **The purpose** of this project is to demostrate knowledge of how to prepare, create, and analyze hypothesis tests.
# 
# **The goal** is to apply descriptive and inferential statistics, probability distributions, and hypothesis testing in Python.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Imports and data loading
# * What data packages will be necessary for hypothesis testing?
# 
# **Part 2:** Conduct hypothesis testing
# * How will descriptive statistics help you analyze your data?
# 
# * How will you formulate your null hypothesis and alternative hypothesis?
# 
# **Part 3:** Communicate insights with stakeholders
# 
# * What key business insight(s) emerge from your hypothesis test?
# 
# * What business recommendations do you propose based on your results?
# 
# <br/>
# 

# In[1]:


# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages fir statistical analysis/hypothesis testing
from scipy import stats


# In[2]:


# Load the dataset
data = pd.read_csv("tiktok_dataset.csv")


# **Data Exploration**

# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


# Check for missing values
data.isna().sum()


# In[6]:


# Drop rows with missing values
data = data.dropna(axis=0)


# In[7]:


data.head()


# In[8]:


data.describe()


# In[9]:


# Compute the mean "video_view_count" for each group in "verified_status"
data.groupby("verified_status")["video_view_count"].mean()


# **Hypothesis Testing**

# The goal is to conduct a two-sample t-test.
# 
# 1. State the null hypothesis and the alternative hypothesis
# 2. Choose a significance level
# 3. Find the p-value
# 4. Reject or fail to reject the null hypothesis
# 
# **$H_O$**: There is no difference in number of views between TikTok videos posted by verified accounts and TikTok videos posted by unverified accounts (any observed difference in the sample data is due to chance or sampling variability).
# 
# **$H_A$**: There is a difference in number of views between TikTok videos posted by verified accounts and TikTok videos posted by unverified accounts (any observed difference in the sample data is due to an actual difference in the corresponding population means).
# 
# Significance level is 5%
# 

# In[11]:


# Two-sample t-test

not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

stats.ttest_ind(a=not_verified, b=verified, equal_var=False)


# p-value < significance level
# 
# Null hypothesis ($H_O$) is rejected.
# 
# There is a statistically signifficant difference in the mean video view count between verified and unverified accounts on TikTok.

# **Insights**

# The analysis shows that there is a statistically significant difference in the average view counts between videos from verified accounts and videos from unverified accounts. This suggests there might be fundamental behavioral differences between these two groups of accounts.
# 
# It would be interesting to investigate the root cause of this behavioral difference. For example, do unverified accounts tend to post more clickbait-y videos? Or are unverified accounts associated with spam bots that help inflate view counts?
# 
# The next step will be to build a regression model on verified_status. A regression model is the natural next step because the end goal is to make predictions on claim status. A regression model for verified_status can help analyze user behavior in this group of verified users. Technical note to prepare regression model: because the data is skewed, and there is a significant difference in account types, it will be key to build a logistic regression model.
