from tweepy import OAuthHandler
import tweepy
import random
from collections import Counter, defaultdict
import sys


def rank_tags():
    global list_100
    z = Counter(list_100)
    d = defaultdict(list)
    for tag, count in z.items():
        d[count].append(tag)

    top_3 = sorted(d.keys(), reverse=True)
    if len(top_3) <= 3:
        return [(tag, count) for count in top_3 for tag in sorted(d[count])]
    else:
        return [(tag, count) for count in top_3[:3] for tag in sorted(d[count])]


def output(ranked):
    global count_tweets
    w = open(output_file, 'a')
    w.write('The number of tweets with tags from the beginning: ' + str(count_tweets) + '\n')
    for i in ranked:
        w.write(str(i[0]) + ' : ' + str(i[1]) + '\n')
    w.write('\n')
    w.close()


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        global count_tweets
        global count_tags
        global list_100

        # this method excludes all retweets
        if status.truncated:
            list_of_hashtags = status.extended_tweet['entities']['hashtags']
            if len(list_of_hashtags) > 0:
                count_tweets += 1
                for i in list_of_hashtags:
                    count_tags += 1
                    if len(list_100) < 100:
                        list_100.append(i['text'])
                    else:
                        prob_reject = random.randint(1, count_tags)
                        if prob_reject <= 100:
                            index_out = random.randint(0, 99)
                            list_100[index_out] = i['text']

                ranked = rank_tags()
                output(ranked)

            """print('Length of list:' + str(len(list_100)))
            print('Count of tags: ' + str(count_tags))
            print('count of tweets: ' + str(count_tweets))"""


# --- EXECUTION STARTS ---
# Read terminal input
port = sys.argv[1]
output_file = sys.argv[2]

w = open(output_file, 'w')
w.close()

auth = OAuthHandler('4whIrM1rCHoiwrkNz7F0fzgCs', 'LvdEAMAb6iFQFK46TwZDFCGvJxzI5lMeaRM7HnNeMrlaxxoA7n')
auth.set_access_token('1276038237585149952-CxAyAR4KNsSvvzrutGqcqmfqe6wJP6', '0WN8WaWZOzJ6kB9n84FUgF3eCCcDIwKPyVxr2onHqPQ8d')

api = tweepy.API(auth)

count_tweets = 0
count_tags = 0
list_100 = []

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener, tweet_mode='extended')
myStream.filter(track=['trump'], languages=['en'])