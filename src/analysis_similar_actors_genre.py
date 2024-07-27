'''
PART 2: SIMILAR ACTROS BY GENRE
Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

#Write your code below
import numpy as np
import pandas as pd
import networkx as nx
import json
from datetime import datetime
from sklearn.metrics import DistanceMetric, pairwise_distances

# Build the graph
g = nx.Graph()

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'
with open('C:/Users/ffatusi1/Desktop/INST414/problem-set-1/imbd_movies.json', 'r') as in_file:
    # Don't forget to comment your code
    for line in in_file:
        # Don't forget to include docstrings for all functions

        # Load the movie from this line
        this_movie = json.loads(line)
        
        # Create a node for every actor
        for actor_id, actor_name in this_movie['actors']:
            g.add_node(actor_id, name=actor_name)
        
        # Iterate through the list of actors, generating all pairs
        ## Starting with the first actor in the list, generate pairs with all subsequent actors
        ## then continue to the second actor in the list and repeat
        
        i = 0  # counter
        for left_actor_id, left_actor_name in this_movie['actors']:
            for right_actor_id, right_actor_name in this_movie['actors'][i+1:]:
                # Get the current weight, if it exists
                if g.has_edge(left_actor_id, right_actor_id):
                    g[left_actor_id][right_actor_id]['weight'] += 1
                else:
                    # Add an edge for these actors
                    g.add_edge(left_actor_id, right_actor_id, weight=1)
            i += 1 

# Print the info below
print("Nodes:", len(g.nodes))

# Calculate centrality metrics
degree_centrality = nx.degree_centrality(g)


# Prepare data for output
data = []
for node in g.nodes(data=True):
    actor_id = node[0]
    actor_name = node[1]['name']
    data.append({
        'actor_id': actor_id,
        'actor_name': actor_name,
        'degree_centrality': degree_centrality[actor_id]
    })

# Convert to DataFrame
centrality_df = pd.DataFrame(data)

# Print the 10 most central nodes by degree centrality
print(centrality_df.sort_values(by='degree_centrality', ascending=False).head(10))

# Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
output_path = f'./network_centrality_{current_datetime}.csv'
centrality_df.to_csv(output_path, index=False)
print(f"Centrality metrics saved to {output_path}")

# PART 2: SIMILAR ACTORS BY GENRE

# Create a data frame where each row corresponds to an actor and each column represents a genre
actor_genre_dict = {}
with open('C:/Users/ffatusi1/Desktop/INST414/problem-set-1/imbd_movies.json', 'r') as in_file:
    for line in in_file:
        this_movie = json.loads(line)
        genres = this_movie['genres']
        for actor_id, actor_name in this_movie['actors']:
            if actor_id not in actor_genre_dict:
                actor_genre_dict[actor_id] = {'actor_name': actor_name}
            for genre in genres:
                if genre not in actor_genre_dict[actor_id]:
                    actor_genre_dict[actor_id][genre] = 0
                actor_genre_dict[actor_id][genre] += 1

actor_genre_df = pd.DataFrame.from_dict(actor_genre_dict, orient='index').fillna(0)

# Select Chris Hemsworth as the query actor
query_actor_id = 'nm1165110'
query_vector = actor_genre_df.loc[query_actor_id].values.reshape(1, -1)

# Calculate Euclidean distances
dist = DistanceMetric.get_metric('euclidean')
euclidean_distances = dist.pairwise(actor_genre_df.drop(columns='actor_name').values)
query_distances = euclidean_distances[actor_genre_df.index.get_loc(query_actor_id)]

# Get the top 10 most similar actors by Euclidean distance
top_10_euclidean_indices = np.argsort(query_distances)[1:11]  # Exclude the query actor itself
top_10_euclidean = actor_genre_df.iloc[top_10_euclidean_indices]

# Print the top 10 most similar actors by Euclidean distance
print("Top 10 actors most similar to Chris Hemsworth (Euclidean distance):")
print(top_10_euclidean[['actor_name']])

# Calculate cosine distances
cosine_distances_matrix = pairwise_distances(actor_genre_df.drop(columns='actor_name'), metric='cosine')
query_cosine_distances = cosine_distances_matrix[actor_genre_df.index.get_loc(query_actor_id)]

# Get the top 10 most similar actors by cosine distance
top_10_cosine_indices = np.argsort(query_cosine_distances)[1:11]  # Exclude the query actor itself
top_10_cosine = actor_genre_df.iloc[top_10_cosine_indices]

# Output the top 10 most similar actors by cosine distance to a CSV
cosine_output_path = f'./similar_actors_genre_{current_datetime}.csv'
top_10_cosine[['actor_name']].to_csv(cosine_output_path, index=True, index_label='actor_id')
print(f"Top 10 similar actors by cosine distance saved to {cosine_output_path}")

# Describe how the list changes based on Euclidean distance
print("Differences in the top 10 similar actors based on Euclidean distance and cosine distance:")
print("Euclidean distance:")
print(top_10_euclidean[['actor_name']])
print("Cosine distance:")
print(top_10_cosine[['actor_name']])
