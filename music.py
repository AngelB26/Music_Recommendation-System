from flask import Flask,jsonify
from flask_restful import Api,Resource
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
api = Api(app)

song_df = pd.read_csv('datasets\song_df.csv')
songs_meta = pd.read_csv("datasets\songs_meta.csv")
song_user = pd.read_csv("datasets\song_user.csv")
interactions = pd.read_csv("datasets\interactions.csv")

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        return user_recommendations
    

pr = popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')

class popularity(Resource):
    def get(self,user_id):
        user = user_id
        results1 = pr.recommend(user)
        json_data = results1.to_json(orient='records')
        return jsonify({'Recommendations': json_data})
    
api.add_resource(popularity, "/popularity/<string:user_id>")

#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items
        
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users

    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
        for i in range(0,len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            for j in range(0,len(user_songs)):       
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
        
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        user_songs = self.get_user_items(user)    
        # print("No. of unique songs for the user: %d" % len(user_songs))
        all_songs = self.get_all_items_train_data()
        # print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations
    
    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations
    
ir = item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')

data1 = pd.read_csv('datasets\main_data.csv')

class content_based_py():
    def __init__(self):
        self.train_data = None
        self.main_data = None
    
    def recommend(self, train_data, main_data):
        song_df2 = pd.merge(train_data, main_data.drop_duplicates(['song']), on='song', how='inner')
        nonplaylist = main_data[~main_data['song'].isin(song_df2['song'].values)]
        non=nonplaylist
        recom =  list(song_df2["song"])
        finalRec = []
        for i in recom:
            selected_song = i
            selected_song_vector = song_df2[song_df2['song'] == selected_song][[ 'acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','tempo']].values
            similarity_scores = cosine_similarity(selected_song_vector,non[[ 'acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','tempo']].values).flatten()
            N = 2
            top_N_recommendations =non.iloc[similarity_scores.argsort()[::-1][1:N+1]]['song'].tolist()
            finalRec = finalRec + top_N_recommendations
        outputValues = pd.DataFrame(finalRec,columns=['Song_Name'])
        return outputValues

cr = content_based_py()

class hybrid(Resource):
    def get(self,user_id):
        user = user_id
        results2 = ir.recommend(user)
        final_results = cr.recommend(results2,data1)
        json_data = final_results.to_json(orient='records')
        return jsonify({'Recommendations': json_data})
    
api.add_resource(hybrid, "/hybrid/<string:user_id>")
    
if __name__ == "__main__":
    app.run(debug=True)