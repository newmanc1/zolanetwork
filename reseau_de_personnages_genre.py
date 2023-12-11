#**TODO: Currently, there are 3 instances of 'magic numbers' that are all related to graph drawing. try to fix after testing on different viles
#also need to test the script still
import pandas as pd
import numpy as np
import json
import networkx
import matplotlib.pyplot as plt
from os import listdir

#!pip install bokeh
from bokeh.io import output_notebook, show, save
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
from networkx.algorithms import community

def add_files(directory):
	## pour créer une liste des fichiers
	#file_names = list()
	df_tokens = list()
	df_entities = list()
	df_books = list()
	    
	for file in sorted(listdir(directory)):
	    
	    
	    
	    #if "entities" in file:
	    #    myName = re.sub(myDir,"",file)
	    #    file_names.append(re.sub(".entities","",myName))     
	    
	    if "tokens" in file:
	        df_token = pd.read_csv(directory + file, delimiter="\t")
	        df_tokens.append(df_token)
	    elif "entities" in file: 
	        df_entity = pd.read_csv(directory + file, delimiter="\t")
	        df_entities.append(df_entity)
	    elif "book" in file and "html" not in file:
	        with open (directory + file, "r") as f:
	            df_book = json.load(f)
	        df_books.append(df_book)
	return df_entities, df_tokens, df_books

#first, get a list of all the different characters and the number of times they appear (including coreference)
#threshold is the number of top entities to select (before entities are filtered by PER and proper noun names, so some will be irrelevant/not wanted)
#**could improve to make it so threshold chooses how many top characters you want, not sure how to do that as is right now though
def get_top_entities(df_entities, threshold):
    top_entities_lists = []
    coref_name_dicts = []
    for i in range(len(df_entities)):
        df_entity = df_entities[i]
        top_indices = df_entity['COREF'].value_counts().index
        top_indices_list = top_indices.to_list()
        top_entities_rows = df_entity.loc[df_entity['COREF'].isin(top_indices_list)] 
        top_entities_list = top_entities_rows.drop_duplicates('COREF')       
        top_entities_list = top_entities_list.loc[(top_entities_list['cat'] == 'PER') & (top_entities_list['prop'] == 'PROP')]       
        top_entities_list = top_entities_list.drop_duplicates('text')       
        top_entities_list = top_entities_list.loc[top_entities_list['COREF'].isin(top_indices_list[0:threshold])]
        coref_name_dict = top_entities_list.set_index('COREF').to_dict()['text']
        top_entities_lists.append(top_entities_list)
        coref_name_dicts.append(coref_name_dict)
    return top_entities_lists, coref_name_dicts 

#for the purposes of making sure that all characters across all books have the best chance to be predicted to be the same character,
#we want to make sure that entity names match up across books.
#to do this, we will 'fix' our top_entities_list by changing all the names to the exact mention that occurs the most often
# (rather than relying on just the very first character mention to determine a characters name)
def fix_entity_names(df_books, top_entities_lists, coref_name_dicts):
    fixed_name_dicts = []
    for i in range(len(df_books)):
        fixed_name_dict = {}
        df_book = df_books[i]
        top_entities_list = top_entities_lists[i]
        coref_name_dict = coref_name_dicts[i]
        entity_count = 0
        index = 0
        while entity_count < len(top_entities_list):
            curr_char = df_book['characters'][index]
            if curr_char['id'] in coref_name_dict:          
                fixed_name_dict[top_entities_list.loc[top_entities_list['COREF'] == curr_char['id']]['text'].iloc[0]] = curr_char['mentions']['proper'][0]['n'].title()      
                top_entities_list.loc[top_entities_list['COREF'] == curr_char['id'], 'text'] = curr_char['mentions']['proper'][0]['n'].title()
                entity_count += 1
            index+=1
        top_entities_lists[i] = top_entities_list
        coref_name_dicts[i] = top_entities_list.set_index('COREF').to_dict()['text']
        fixed_name_dicts.append(fixed_name_dict)
    return top_entities_lists, coref_name_dicts, fixed_name_dicts
#add booknlp's inferred gender to a dictionary for use in the network
#for those who do not have a strong prediction, we will add the names of the top 2 characters to make
#2 nodes for this character (because in our debugging we found it is likely that 2 characters are mixed together)
def add_gender(df_books, top_entities_lists, coref_name_dicts):
    gender_dict = {} #maps name to inferred gender in .book
    errors_dict = {} #if gender is 'error' (inference 'max' is <.8), put in this dict and map name to 2nd most common mention name, proportion of entity1 count, entity2 count
    for i in range(len(df_books)):
        df_book = df_books[i]
        top_entities_list = top_entities_lists[i]
        coref_name_dict = coref_name_dicts[i]
        entity_count = 0 # to count the number of characters matched so far
        character_index = 0 #to loop thru characters in book
        while entity_count < len(top_entities_list):
            curr_char = df_book['characters'][character_index]
            if curr_char['id'] in coref_name_dict:
                if curr_char['g']['max'] >= .8:
                    gender_dict[coref_name_dict[curr_char['id']]] = curr_char['g']['argmax']
                else:
                    #errors_list[0] = 2nd most common entity name (entity2)
                    #errors_list[1] = proportion of entity1 (entity1 count / entity1 count + entity2 count) * total count
                    #errors_list[2] = proportion of entity2
                    errors_list = []
                    curr_char_prop = curr_char["mentions"]["proper"]
                    if len(curr_char_prop) > 1 and curr_char_prop[0]['n'] not in curr_char_prop[1]['n']:
                        errors_list.append(curr_char_prop[1]['n'])
                        errors_list.append(round(curr_char_prop[0]['c'] / (curr_char_prop[0]['c'] + curr_char_prop[1]['c']) * curr_char["count"], 0))
                        errors_list.append(round(curr_char_prop[1]['c'] / (curr_char_prop[0]['c'] + curr_char_prop[1]['c']) * curr_char["count"], 0))
                        errors_dict[coref_name_dict[curr_char['id']]] = errors_list
                        if curr_char['g']['argmax'] == 'il/le':
                            gender_dict[errors_list[0]] = 'elle/la'
                        else:
                            gender_dict[errors_list[0]] = 'il/le'
                    gender_dict[coref_name_dict[curr_char['id']]] = curr_char['g']['argmax']
                    
                entity_count += 1
            character_index += 1
    return gender_dict, errors_dict

# #make a dictionary that maps each name to a predicted gender as given in .book
# def add_gender(df_books, top_entities_lists, coref_name_dicts):
#     gender_dict = {}
#     for i in range(len(df_books)):
#         df_book = df_books[i]
#         top_entities_list = top_entities_lists[i]
#         coref_name_dict = coref_name_dicts[i]
        
    
#         entity_count = 0 # to count the number of characters matched so far
#         character_index = 0 #to loop thru characters in book
#         while entity_count < len(top_entities_list):
#             curr_char = df_book['characters'][character_index]
#             if curr_char['id'] in coref_name_dict:
#                 if curr_char['g']['max'] >= .8:
#                     gender_dict[coref_name_dict[curr_char['id']]] = curr_char['g']['argmax']
#                 else:
#                     gender_dict[coref_name_dict[curr_char['id']]] = 'error'
#                 entity_count += 1
#             character_index += 1
#     return gender_dict

#now, let's get the number of times each one appears

#top_entities_list is df of most common entities (as returned by get_top_entities)
#df_entities is df with all entities
#returns a dictionary that maps top entity name : number of times it appears (including coreference)
def get_top_entities_counts(top_entities_lists, df_entities, coref_name_dicts):
    top_entities_list_indices_list = []
    top_entities_counts_dicts = []
    for i in range(len(top_entities_lists)):
        
        top_entities_list = top_entities_lists[i]
        df_entity = df_entities[i]
        coref_name_dict = coref_name_dicts[i]
        
        top_entities_list_indices = top_entities_list['COREF'] #all the top COREF numbers
        top_entities_counts = df_entity.loc[df_entity['COREF'].isin(top_entities_list_indices)]
        top_entities_counts = top_entities_counts['COREF'].value_counts()
        
        #counts for the amount of times each top entity appears
        
        top_entities_counts_dict = {}
        for entity_coref, name in coref_name_dict.items():
                
                
            top_entities_counts_dict[name] = top_entities_counts.loc[entity_coref]
                
        top_entities_list_indices_list.append(top_entities_list_indices)
        top_entities_counts_dicts.append(top_entities_counts_dict)
    return top_entities_list_indices_list, top_entities_counts_dicts


#df_entities - overall dataframe with all values
#top_entities_df - df with just the list of top characters, no repeats/coreference
#n - size of window of tokens
#char_dict - character dictionary that counts interactions betewen 2 entities.  *assumed to already have empty dictionaries that account for every link*
# char_dict[entity1][entity2] will be the amount of co-occurrences within windows of n tokens of entity1, entity2
#entity_index - index of top_entities_rows where your given entity is
def get_pers_within_n_tokens(df_tokens, df_entities, top_entities_list, fixed_name_dict, n, char_dict, entity_index, use_fixed_names):
    #using our given entity_index, find the start and end token of the entity
    start_token_index = df_entities.iloc[entity_index]['start_token']
    end_token_index = df_entities.iloc[entity_index]['end_token']
    #keep track of the entity we are starting on
    if use_fixed_names:
        orig_entity = fixed_name_dict[df_entities.loc[df_entities['COREF'] == df_entities.iloc[entity_index]['COREF']].iloc[0]['text']] 
    else:
        orig_entity = df_entities.loc[df_entities['COREF'] == df_entities.iloc[entity_index]['COREF']].iloc[0]['text']
    curr_entity_index = entity_index
    curr_entity = orig_entity
    curr_end_token_index = df_entities.iloc[curr_entity_index]['end_token']
    #starting token index is end_token number of given entity
    #while loop to find entities with end token >= index-n
    while curr_end_token_index >= end_token_index-n and curr_end_token_index >= 0:
        if curr_entity in fixed_name_dict:
            curr_entity = fixed_name_dict[curr_entity]
        #check to see if the entity we are on is not our original entity, and if it's in the list of top entities
        if curr_entity != orig_entity and curr_entity in char_dict:
            char_dict[orig_entity][curr_entity] += 1
        if curr_entity_index == 0:
            break
        curr_entity_index -= 1
        curr_end_token_index = df_entities.iloc[curr_entity_index]['end_token']
        #find the end token of the previous entity
        curr_entity = df_entities.loc[df_entities['COREF'] == df_entities.iloc[curr_entity_index]['COREF']].iloc[0]['text']
        #update the entity name of curr entity. this long expression gives us the original name of the entity using its first appearance, 
        #so it works even if it is a coreference
    #now, do a while loop for entities with start token <= index+n
    curr_entity_index = entity_index
    curr_start_token_index = df_entities.iloc[curr_entity_index]['start_token']
    #starting token index is start_token number of given entity
    curr_entity = orig_entity
    while curr_start_token_index <= start_token_index+n and curr_start_token_index < len(df_tokens):
        if curr_entity in fixed_name_dict:
            curr_entity = fixed_name_dict[curr_entity]
        if curr_entity != orig_entity and curr_entity in char_dict:
            char_dict[orig_entity][curr_entity] += 1

        curr_entity_index += 1
        if curr_entity_index >= len(df_entities):
            break
        curr_start_token_index = df_entities.iloc[curr_entity_index]['start_token']
        curr_entity = df_entities.loc[df_entities['COREF'] == df_entities.iloc[curr_entity_index]['COREF']].iloc[0]['text']

    return char_dict

#create dictionary for each possible link between characters
def create_characters_dict(top_entities_lists):
    characters_dicts = []
    for top_entities_list in top_entities_lists:
        characters_dict = {}
        for entity1 in top_entities_list['text']:
            characters_dict[entity1] = {}
            for entity2 in top_entities_list['text']:
                if entity1 != entity2:
                    characters_dict[entity1][entity2] = 0
        characters_dicts.append(characters_dict)
    return characters_dicts


#this method will both add the size/weight of each node as well as colors to distinguish the importance of characters
#threshold 1 will be the index of the smallest node you want to be the first color
#e.g. with threshold1 = 2, nodes 0, 1, 2, the 3 largest nodes, would all be color 1
#nodes 3 - threshold2 would be color 2, and nodes threshold2 - end would be color 3
#threshold can either be set to index or percentage
#for index, we would count x number of nodes to be a color
#for percentage, the top x% of nodes would be a color
def add_nodes(G, threshold1, threshold2, top_entities_counts_dict, gender_dicts, characters_dicts, threshold='percentage', color1='lightblue', color2='darksalmon', color3='palegoldenrod'):
    node_sizes = []
    color_map = [] 
    color_dict = {}
    novels_dict = {} #list of all the romans this character appears in
    book_dict = {1: "La Fortune des Rougon", 2: "La Curée", 3: "Le Ventre de Paris", 4: "La Conquête de Plassans", 5: "La Faute de l'abbé Mouret", 
            6: "Son Excellence Eugène Rougon", 7: "L'Assommoir", 8: "Une page d'amour", 9: "Nana", 10: "Pot-Bouille", 
            11: "Au Bonheur des Dames", 12: "La Joie de vivre", 13: "Germinal", 14: "L'Œuvre", 15: "La Terre",
            16: "Le Rêve", 17: "La Bête humaine", 18: "L'Argent", 19: "La Débâcle", 20: "Le Docteur Pascal"}

    sorted_nodes = sorted([top_entities_counts_dict[node] for node in G.nodes()], reverse=True)
    percentile1 = np.percentile(sorted_nodes, threshold1)
    percentile2 = np.percentile(sorted_nodes, threshold2)
    
    for node in G.nodes():
        size = top_entities_counts_dict[node]
        novels = ""
        
        gender = gender_dicts[node]
        if gender == 'il/le':
            color_map.append(color1)
            color_dict[node] = color1
            node_sizes.append(np.log(size) ** 3 * 5)
        elif gender == 'elle/la':
            color_map.append(color2)
            color_dict[node] = color2
            node_sizes.append(np.log(size) ** 3 * 5)
        else:
            
            color_map.append(color3)
            color_dict[node] = color3
            node_sizes.append(np.log(size) ** 3 * 5)
            
            
        for i in range(len(characters_dicts)):
            character_dict = characters_dicts[i]
            if node in character_dict:
                novels += book_dict[i + 1] + ", "
            
        
        novels_dict[node] = novels[:-2]
    
    return node_sizes, color_map, color_dict, novels_dict



#includes node labels and edge weight labels
def draw_basic_graph(G, pos):
	
	networkx.draw(G, with_labels=True, node_size=100)
	labels = networkx.get_edge_attributes(G,'weight')
	networkx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
	plt.show()

def draw_colored_graph(G, pos, threshold1, threshold2, threshold_type, top_entities_counts_dict, gender_dict):
	#by count
	node_sizes, color_map, color_dict = add_nodes(G, threshold1, threshold2, top_entities_counts_dict, gender_dict, threshold_type)
	#by percentage
	#node_sizes, color_map = add_nodes(G.nodes(), 95, 70, threshold='percentage')
	edges = G.edges()
	max_edge = max([G[u][v]['weight'] for u,v in edges])

	#**TODO: if possible, fix this magic number for weights
	weights = [(G[u][v]['weight'] / max_edge) * 30 for u,v in edges] #normalizing a bit, currently arbitrary
	plt.figure(2,figsize=(17,17)) 
	networkx.draw(G, pos, width=weights,with_labels=True, node_size=node_sizes, node_color=color_map)
	plt.savefig("colored_graph.png")
	

def draw_interactive_graph(G, pos, threshold1, threshold2, threshold_type, top_entities_counts_dict, gender_dict, characters_dicts, title):
	#by count
    node_sizes, color_map, color_dict, novels_dict = add_nodes(G, threshold1, threshold2, top_entities_counts_dict, gender_dict, characters_dicts, threshold_type)
	#by percentage
	#node_sizes, color_map = add_nodes(G.nodes(), 95, 70, threshold='percentage')
    edges = G.edges()
    max_edge = max([G[u][v]['weight'] for u,v in edges])
    degrees = dict(networkx.degree(G))
    networkx.set_node_attributes(G, name='degree', values=degrees)
    number_to_adjust_by = 5
    adjusted_top_entities_counts_dict = dict([(node, (np.log(size)**2) + 3) for node, size in top_entities_counts_dict.items()])
    adjusted_node_size = adjusted_top_entities_counts_dict #dict([(node, degree+number_to_adjust_by) for node, degree in networkx.degree(G)])
    networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
    networkx.set_node_attributes(G, name='node_size', values=top_entities_counts_dict)
    names = {}
    for name in list(G.nodes()):
        names[name] = name #jank solution but it works
    networkx.set_node_attributes(G, name='name', values=names) 
    color = color_dict
    networkx.set_node_attributes(G, name='color', values=color)
    appearances = novels_dict
    networkx.set_node_attributes(G, name='appearances', values=appearances)
    networkx.set_node_attributes(G, name='betweenness', values=networkx.betweenness_centrality(G))
    networkx.set_node_attributes(G, name='closeness', values=networkx.closeness_centrality(G))


    mapping = dict((n, i) for i, n in enumerate(G.nodes))
    H = networkx.relabel_nodes(G, mapping)
    #Choose colors for node and edge highlighting
    node_highlight_color = 'white'
    edge_highlight_color = 'black'

	#Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'color'

	#Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
    color_palette = Blues8

	

	#Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [
           ("Character", "@name"),
            ("Degree", "@degree"),
            ("Size", "@node_size"),
            ("Color", "@color"),
            ("Appearances", "@appearances"),
            ("Betweenness", "@betweenness"),
            ("Closeness", "@closeness")
	        # ("Modularity Class", "@modularity_class"),
	       # ("Modularity Color", "$color[swatch]:modularity_color"),
    ]

	#Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = HOVER_TOOLTIPS,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

	#Create a network graph object
	# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(H, networkx.spring_layout, scale=10, center=(0, 0))

	#Set node sizes and colors according to node degree (color as category from attribute)
    network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
	#Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
    network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

	#Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.3, line_width=1)
	#Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
	#**TODO: if possible, fix this magic number for edges
    network_graph.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a,b)['weight'] / max_edge * 30 for a, b in G.edges()]
    network_graph.edge_renderer.glyph.line_width = {'field': 'line_width'}


	#Highlight nodes and edges
    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(network_graph)

	#Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
    plot.renderers.append(labels)

    show(plot)
    save(plot, filename=f"{title}.html")


#file_name = name of file (ex: 1883_Guy-de-Maupassant_Une-vie)
#N_pers = number of people you want in the réseau
#N_tokens = length of window around each occurrence of an entity when calculating co-occurrences
def main(directory_name, use_fixed_names, N_pers, N_tokens, graph_type, threshold1=97, threshold2=85, threshold_type='percentage'):
    df_entities, df_tokens, df_books = add_files(directory_name)

#**prob make 1 big for loop
    for df_entity in df_entities:
	
        df_entity['text'] = df_entity['text'].map(lambda x: x.lstrip('* .').rstrip('* .'))
	#clean up the trailing spaces, asterisks, periods for each token
    top_entities_lists, coref_name_dicts = get_top_entities(df_entities, 20)
    fixed_name_dicts = {}   

    if use_fixed_names:
        top_entities_lists, coref_name_dicts, fixed_name_dicts = fix_entity_names(df_books, top_entities_lists, coref_name_dicts)
    top_entities_list_indices_list, top_entities_counts_dicts = get_top_entities_counts(top_entities_lists, df_entities, coref_name_dicts)
    gender_dict, errors_dict = add_gender(df_books, top_entities_lists, coref_name_dicts)
    #now that we have this list, we want to find every instance where each character appears, and put windows of 
    #+- N tokens around each
    characters_dicts = create_characters_dict(top_entities_lists)
    top_entities_rows_list = []
    top_entities_rows_list = []
    for i in range(len(df_entities)):
        df_entity = df_entities[i]
        top_entities_list_indices = top_entities_list_indices_list[i]
        top_entities_rows = df_entity.loc[df_entity['COREF'].isin(top_entities_list_indices)] #gives us all rows where a top entity appears
        top_entities_rows_list.append(top_entities_rows)
    for i in range(len(top_entities_rows_list)):
        
        df_token = df_tokens[i]
        df_entity = df_entities[i]
        top_entities_list = top_entities_lists[i]
        characters_dict = characters_dicts[i]
        fixed_name_dict = {}
        if use_fixed_names:
            fixed_name_dict = fixed_name_dicts[i]
        top_entities_rows_list[i].apply(lambda x: get_pers_within_n_tokens(df_token, df_entity, top_entities_list, fixed_name_dict, 20, characters_dict, x.name, True), axis=1)   
    #now that we have a full dictionary, we want to turn that into a dataframe and then a network. we will also use this opportunity
    #to duplicate the rows to fix for the characters in error_list
    interactions_dfs = []
    error_set = set(errors_dict)
    for i in range(len(characters_dicts)):
        characters_dict = characters_dicts[i]
        
        #we need to loop thru all characters_dicts and find all characters that are in errors_dict
        for name in error_set.intersection(characters_dict):
            
            #now, we want to create a duplicate row for the character in error_list
            #for example, for silvere/miette we want to make a duplicate row of silvere for miette
            error_list = errors_dict[name]
            characters_dict[error_list[0]] = characters_dict[name].copy()
            #afterwards, we need to update every single entry in characters_dict to add dict[i][miette], which will be the same value as for silvere
            #so loop thru every other character in the dict
            for char, d in characters_dict.items():
                if char != name and char != error_list[0]:
                    
                    d[error_list[0]] = d[name]
      
        interactions_df = pd.DataFrame.from_dict(characters_dict)
        interactions_dfs.append(interactions_df)

	#merge interactions dataframes

    merged_interactions_df = pd.concat(interactions_dfs, join='outer', axis=1)

    merged_interactions_df = merged_interactions_df.groupby(level=0, axis=1).sum()

    #merge top_entities_counts dictionaries
    merged_top_entities_counts_dict = {}
    for d in top_entities_counts_dicts:
        for entity, count in d.items():
            merged_top_entities_counts_dict[entity] = merged_top_entities_counts_dict.get(entity, 0) + count
    for name, error_list in errors_dict.items():

        if error_list[1] < merged_top_entities_counts_dict[name] * .4:

            
            #unelegant solution, which accounts for entities that have errors with multiple novels, in which they end up getting shrunk
            #because the program only takes account of the proportion for errors for the latest appearance in a novel,
            #which could end up getting rid of a lot of its mentions.
            merged_top_entities_counts_dict[name] += error_list[1]
        else:
            merged_top_entities_counts_dict[name] = error_list[1]
        if error_list[0] in merged_top_entities_counts_dict:
            merged_top_entities_counts_dict[error_list[0]] += error_list[2]
            
        else:
            merged_top_entities_counts_dict[error_list[0]] = error_list[2]
    #now let's make the network
    G=networkx.from_pandas_adjacency(merged_interactions_df)
    #remove edges from a character to themself
    for entity in merged_top_entities_counts_dict.keys():
        if G.has_edge(entity, entity):
            G.remove_edge(entity, entity)
    pos = networkx.spring_layout(G)
    if graph_type == 'basic':
        draw_basic_graph(G, pos)
    elif graph_type == 'colored':
        draw_colored_graph(G, pos, threshold1, threshold2, threshold_type, merged_top_entities_counts_dict, gender_dict)
    elif graph_type == 'interactive':
        draw_interactive_graph(G, pos, threshold1, threshold2, threshold_type, merged_top_entities_counts_dict, gender_dict, characters_dicts, 'Gendered Zola networks œuvres 1-' + str(len(df_tokens)))
	

    
if __name__ == "__main__":
	main("OUTPUT_ZOLA_TEST/", True, 35, 20, "interactive")




