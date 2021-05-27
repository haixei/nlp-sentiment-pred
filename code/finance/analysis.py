import pandas as pd
import plotly.express as pltx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
import string
from collections import Counter

data = pd.read_csv('../../data/financial-news.csv')
data.columns = ['sentiment', 'headline']
print(data.head())

# Explore data structure
sent_hist_fig = pltx.histogram(data, x='sentiment', title='Sentiment distribution in the data set', color='sentiment', color_discrete_sequence=['#ffbda1', '#5a499c', '#e85a77'])
# >> sent_hist_fig.show()

v_counts_sent = data['sentiment'].value_counts()
for sentiment, value in v_counts_sent.items():
    percent = round((value/data.shape[0]) * 100, 2)
    print(sentiment, ':', str(percent) + '%')


# Plot values
# Get lenght of the headlines trough the dataset
data['num_of_words'] = data['headline'].str.split()

# Count the words and remove these shorter than 2 letters
data['num_of_words'] = data['num_of_words'].apply(lambda x: [word for word in x if len(word) > 1])
data['num_of_words'] = data['num_of_words'].apply(lambda x: len(x))

print(data.head())

# Show the distribution of headline length
words_hist_fig = pltx.histogram(data, x='num_of_words', title='Headline length distribution', color_discrete_sequence=['#5a499c'])
# >> words_hist_fig.show()

# Show the amount of punctuation and words by sentiment
sent_word_hist_fig = make_subplots(rows=1, cols=3)
print(v_counts_sent.index[0])
sent_colors = {'positive': '#e85a77', 'negative': '#5a499c', 'neutral': '#ffbda1'}
i = 1
for sentiment, value in sent_colors.items():
    data_sent = data.loc[data['sentiment'] == sentiment]
    sent_word_hist_fig.add_trace(
        go.Histogram(x=data_sent['num_of_words'], name=sentiment, marker=dict(color=sent_colors[sentiment])),
        row=1, col=i
    )
    i += 1

sent_word_hist_fig.update_layout(title_text="Distribution of words by sentiment")
sent_word_hist_fig.update_traces(opacity=0.75)
# >> sent_word_hist_fig.show()

# Amount of stopwords
stop_words = set(stopwords.words('english'))
data['num_of_stopwords'] = data['headline'].apply(lambda x: len(set(x.split()) & stop_words))

# Save the different properties of headers
# Amount of punctuation
data['num_of_punct'] = data['headline'].str.split()
count_punct = lambda l1, l2: sum([1 for x in l1 if x in l2])
data['num_of_punct'] = data['num_of_punct'].apply(lambda x: count_punct(x, set(string.punctuation)))

# Amount of characters
data['num_of_char'] = data['headline'].apply(lambda x: len(x))

print(data.head())

# Correlation heatmap
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='Sunsetdark')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()

box_fig = pltx.box(data, x="sentiment", y='num_of_char', points="all", color_discrete_sequence=['#de439b'])
# >> box_fig.show()

# Amount of unique words in the dataset
data_words = data['headline'].str.lower().str.split()
results = Counter()
data_words.apply(results.update)

# Remove words with less than 4 letters
filtered_words = {k: v for k, v in results.items() if len(k) > 3}
best_ten = dict(Counter(filtered_words).most_common(10))

for_plot = {'word': [], 'amount': []}
for key, val in best_ten.items():
    for_plot['word'].append(key)
    for_plot['amount'].append(val)

line_fig = pltx.line(for_plot, x='word', y='amount', title='Top words in the data set', color_discrete_sequence=['#5a499c'])
# >> line_fig.show()
