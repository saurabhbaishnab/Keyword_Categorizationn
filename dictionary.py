import sys
import pandas as pd
import enchant
import xlrd
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sklearn
from sklearn.naive_bayes import BernoulliNB
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

d = enchant.Dict("en_US")
stop_words = set(stopwords.words("english"))
#print(type(stop_words))
stop_words.add('*')
ps = PorterStemmer()

#file_name = "C:/Users/saurabh453/input/demo.csv"
file_name = "./PrintCategoryData.csv"
supplies = pd.read_csv(file_name, encoding = "")
keyword_trans_col = supplies[['Keyword', 'Translation']].copy()
rows_list = []
for index, row in keyword_trans_col.iterrows():
		
	keyword_item = row['Keyword']
	Translation_item = row['Translation']

	if type(row['Keyword']) == str:
		
		if type(Translation_item) == str and  d.check((keyword_item)) == False :
			item = Translation_item
			#rows_list.append(Translation_item)
		else:
			item = keyword_item
			#rows_list.append(keyword_item)
			
	else:
		item = keyword_item
	new_word = ""
	words = word_tokenize(item)
	words_length = len(words)
	
	for w in words:
	
		if w not in stop_words:
			stemmed_w = ps.stem(w)
			
			if(words_length == 1):
				new_word += stemmed_w
				
			else:
				new_word += stemmed_w + " "
				words_length = words_length - 1
			
	rows_list.append(new_word)

df1 = pd.DataFrame(rows_list, columns = ['Keyword'])

category_col = pd.DataFrame({'Category': supplies['Category']})

frames = [df1, category_col]
result = pd.DataFrame({'Keyword' : rows_list, 'Category': supplies['Category']})
result_drop = result.drop_duplicates(subset=['Keyword', 'Category'], keep=False)

result.Category = result.Category.str.replace(' ', '')

#print(result)


# for item in result['Category']:
#     result.drop(result[type(item) != str].index, axis = 0, inplace=True)
        
locations =  result[result['Category'].apply(lambda x: type(x)!=str)]

#location = result[type(result['Category']) == str].index
# print(type(locations))
# print(locations)
list(locations.columns.values)
index_list = []
index_list = locations.index.values
#print(index_list)
result.drop(index_list, inplace=True)
keyword_dict = {}
# for item in result['Category']:
#     if type(item) != str:
#         print(item)


keyword_train, keyword_test, category_train, category_test = train_test_split(result['Keyword'],result['Category'], test_size = .33, random_state= 17)



#print(category_train.unique())
#BerNB = BernoulliNB(binarize = True)
#BerNB.fit(keyword_train, category_train)

#category_expect = category_test
#category_pred = BerNB.predict(keyword_test)

#print(accuracy_score(category_expect, category_pred))

# j =0
# splitted_item = []
# for row in result['Keyword'].unique():
# 	for keyword_elements in (row.split(" ")):
# 		if keyword_elements not in keyword_dict.keys():
# 			keyword_dict[keyword_elements] = j
# 			j = j+1
# #print(keyword_dict)

# a = []
# for item in result['Keyword']:
# 	b = []
# 	for i in (item.split(" ")):
# 		b.append(keyword_dict[i])
# 	a.append(b)
# #print(a)
# Category_list = []
# for row in result['Category']:
# 	Category_list.append(row)
#le = preprocessing.LabelEncoder()
#result['Keyword'] = le.fit_transform(result['Keyword'].astype(str))
#result['Category'] = le.fit_transform(result['Category'].astype(str))





from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit_transform(keyword_train)
count_vect.get_feature_names()
X_train = count_vect.transform(keyword_train)
print(X_train)





X_test = count_vect.transform(keyword_test)
X_test



from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,category_train.values)


y_pred = nb.predict(X_test)

from sklearn import metrics
metrics.accuracy_score(category_test, y_pred)
