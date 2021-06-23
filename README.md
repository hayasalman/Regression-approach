# Regression-approach
#create new column called intercept and new column for get_dummies() of pages
df2['intercept'] = 1
df2[['new_page', 'old_page']] = pd.get_dummies(df2['landing_page'])
df2[['control', 'ab_page']] = pd.get_dummies(df2['group'])
#view head to ensure the new columns applied 
df2.head()
#use drop() to remove any unnecessary columns
df2.drop(['new_page', 'control' ], axis=1, inplace=True)
df2.head()
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#use logistic model
logit_reg = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = logit_reg.fit()
results.summary()
#we take the exponentiate the coefficients
1/np.exp(-0.0150)
#load countries CSV file
countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
#the number of unique countries in this dstaset
df_new['country'].unique()
### Create the necessary dummy variables
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()
#use logistic model# drop US 
logit_reg = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK']])
results = logit_reg.fit()
results.summary()
#we take the exponentiate the coefficients
1/np.exp(-0.0408), np.exp(0.0099)
### Fit Your Linear Model And Obtain the Results
df_new['UK_new_page'] = df_new['UK']*df_new['ab_page']
df_new['US_new_page'] = df_new['US']*df_new['ab_page']
df_new['CA_new_page'] = df_new['CA']*df_new['ab_page']
df_new.head()
#use logistic model
logit_reg = sm.Logit(df_new['converted'], df_new[['intercept','CA','UK','CA_new_page', 'UK_new_page']])
results = logit_reg.fit()
results.summary()
#we take the exponentiate the coefficients
1/np.exp(-0.0073), np.exp(0.0045)
#we take the exponentiate the coefficients
1/np.exp(-0.0674), np.exp(0.0108)
