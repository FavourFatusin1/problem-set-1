'''
PART 1: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Guild a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to. 
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is line with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json
from datetime import datetime

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
output_path = f'C:/Users/ffatusi1/Desktop/INST414/problem-set-1/network_centrality_{current_datetime}.csv'
centrality_df.to_csv(output_path, index=False)
print(f"Centrality metrics saved to {output_path}")
