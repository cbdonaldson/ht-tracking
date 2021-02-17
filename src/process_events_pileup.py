#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd 
import scipy.ndimage as nd
import time
import pickle
import random
import os
import math
from skimage.feature import blob_dog, blob_log, blob_doh
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Queue
from skimage.color import rgb2gray
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from decoder import decode_hit
from tracking_functions import read_clusters, correct_layers, find_dog_peaks, box_filter


#if len(sys.argv) < 3 :
#    print('Usage: python process_events.py <data_dir> <event_start> <event_end>'
#        + '<iterations> <t1> <t2> <t_match> <nn> ')
#    sys.exit(0)
#
#print('Reading in cluster files and pattern banks...', flush=True)

###########
## SETUP ##
###########

data_dir = sys.argv[1]
event_num = int(sys.argv[2])
iterations = int(sys.argv[3])
size = int(sys.argv[4])
t1 = int(sys.argv[5])
t2 = float(sys.argv[6])
t_match = float(sys.argv[7])
nn = int(sys.argv[8])

#event_nums = [e for e in range(event_start, event_end+1)]

pattern_datatype = []
pattern_datatype.append(('dict_index', 'i4'))
pattern_datatype.append(('pattern_index','i4'))
for layer in range(0,8):
    pattern_datatype.append(('hit_L' + str(layer),'uint16'))

dict_datatype = []
dict_datatype.append(('sector_index', 'i4'))
for layer in range(0,8):
    dict_datatype.append(('hit_L' + str(layer),'i4'))

mean_datatype = []
mean_datatype.append(('dict_index', 'i4'))
mean_datatype.append(('col', 'int8'))
mean_datatype.append(('pix_row', 'int8'))
for layer in range(7):
    mean_datatype.append(('hit_L' + str(layer+1), 'int8'))

w_dt = []
w_dt.append(('sector_index', 'i4'))
w_dt.append(('col', 'u1'))
for layer in range(0,8):
    w_dt.append(('L' + str(layer),'u1'))


pattern_records = np.fromfile(data_dir + 'reduced_patterns.bin', pattern_datatype)
pattern_df = pd.DataFrame(pattern_records)

dict_records = np.fromfile(data_dir + 'sector_dictionary.bin', dict_datatype)
dict_df = pd.DataFrame(dict_records)

compressed_info = np.fromfile(data_dir + 'compressed_info.bin', mean_datatype)
compressed_df = pd.DataFrame(compressed_info)

popular_sector_widths = np.fromfile(data_dir + 'popular_sector_widths.bin', w_dt)
dc_info = pd.DataFrame(popular_sector_widths)

layer_element_list = pickle.load( open( "../storage/layer_element_list.p", "rb" ) )

maxStripP = np.uint16(21)
maxCol = np.uint16(1)

clusters = read_clusters(data_dir + 'hit_events_pileup_200_eta0.1-0.3_200events.bin')
#clusters = read_clusters(data_dir + 'hit_events_no_pileup_eta0.1-0.3_200events.bin')

#complete_sector_ids = dict_df.loc[dict_df.isin([-1]).sum(1) == 0].sector_index.values
sector_ids = dict_df.sector_index.values
popular_sectors = []
#for sector in complete_sector_ids:
#    n = len(pattern_df[pattern_df.dict_index == sector])
#    if n >= 30:
#        popular_sectors.append(sector)
for sector in sector_ids:
    n = len(pattern_df[pattern_df.dict_index == sector])
    if n >= 30:
        popular_sectors.append(sector)

######################
## Define functions ##
######################


def get_bresenham_line(x1, y1, x2, y2, width = 1) :
    im = Image.new('1', (size, size), 0)
    draw = ImageDraw.Draw(im)
    draw.line((x1, y1, x2, y2), fill=(1), width = width)
    pixels_x, pixels_y = np.where(np.array(im) == True)
    return (pixels_x, pixels_y)

def event_HT_tracking(event_num):

    """
    Processes a range of full events to produce an output Hough image with
    localised track recognition.

    """

    print('Event', event_num)

    detector_elements = []
    sectors = []
    found_sectors = []
    coll_indicies = []

    event = clusters[event_num]
    collections = event.colls

    ## Find all fired detector_elements in the event

    for coll_i in range(len(collections)):
        coll = collections[coll_i]
        hashId = coll.hashId
        detector_elements.append([coll_i, hashId])

    de_df = pd.DataFrame(detector_elements, columns=['coll_i', 'hashId'])
 
    ## Loop through popular sectors and store the sector/sectorId if
    ## at least 7 of the hashIds contain hits

    for sector_index in popular_sectors:
        colls = []
        full_sector = dict_df[dict_df.sector_index == (sector_index)].values[0][:]
        sector = full_sector[1:]
        #if full_sector[0] in matched_sectors:
        #if -1 in sector:
        #    pass
        #else:
        counter = 0
        for hashId in sector:
            if hashId in de_df.hashId.values:
                coll_index = de_df[de_df.hashId == hashId].values[0][0]
                colls.append(coll_index)
                counter += 1
        if counter >= 6:
            found_sectors.append(sector_index)
            sectors.append(sector)
            coll_indicies.append(colls)

    #print(found_sectors)

    ## Consider each found_sector for tracking

    event_results = []

    for num in range(len(found_sectors)):

        
        time0 = time.time()

        print('Sector', num+1, 'of', len(found_sectors))

        hits = []
        columns = []
        hashIds = []
        muons = []
        hit_set = []

        sector_index = found_sectors[num]
        
#        if sector_index != cs:
#            continue
        
        print('sector index', sector_index, flush=True)
        colls = coll_indicies[num]
        layer_widths_full = dc_info[dc_info.sector_index == sector_index]
        
        for coll_i in colls:
            coll = collections[coll_i]
            if coll.hashId in sectors[num]:
                for hit in coll.hits:

                    hit_set.append(hit)
                    hit_info = decode_hit(hit[0], maxStripP)
                    hits.append(hit_info[0])
                    hashIds.append(coll.hashId)
                    if hit_info[1] != -1:
                        columns.append(hit_info[1])

        ## need the condition here that there is at least one hit in 7 different layers, otherwise no point proceeding.
        ## allocate each hit to a layer

 #       print('number of hits', len(hits))
        
        full_hit_set = []
        layer_list = []
        for i in range(len(hashIds)):
            for layer_num in range(len(layer_element_list)):
                if hashIds[i] in layer_element_list[layer_num]:
                    layer_list.append(layer_num)
                    full_hit_set.append([layer_num, hit_set[i]])
                    

        #if len(hits) == 7:
        #        correct_layers(hits, layer_list)

        sector_info = compressed_df[compressed_df.dict_index == sector_index].drop('dict_index', axis=1)

        ## according to the column measurement

        if len(columns) > 0:
            col = columns[0]
            col_info = sector_info[sector_info.col == col].drop('col', axis=1)


            if len(col_info) > 0:

                layer_widths_col = layer_widths_full[layer_widths_full.col == col].drop(
                'col', axis=1).drop('sector_index', axis=1).values[0]

                mean_pattern = col_info.iloc[0,:].values
                eigen1 = col_info.iloc[1,:].values
                eigen2 = col_info.iloc[2,:].values
                eigen1_true = [x/100 for x in eigen1]
                eigen2_true = [x/100 for x in eigen2]
                eigenpatterns = [eigen1_true, eigen2_true]

                #histo, lines, votes = hough_transform_PIL(hits, layer_list, mean_pattern, eigenpatterns, layer_widths_col)
                histo, lines, votes, hit_map = hit_array_HT(hits, full_hit_set, layer_list,
                                                            mean_pattern, eigenpatterns, layer_widths_col)


                for i in range(iterations):
                    histo, votes, hit_map = update_votes(histo, lines, votes, hit_map, t1, t2)
                        #print('iteration = ', i)

                #plt.imshow(histo)
                #plt.colorbar()
                        
                #print('max = ' + str(histo.max()))
                event_results.append([event_num, sector_index, histo, hit_map])

            else:

                col = int(not col)
                col_info = sector_info[sector_info.col == col].drop('col', axis=1)

                if len(col_info) > 0:


                    layer_widths_col = layer_widths_full[layer_widths_full.col == col].drop(
                    'col', axis=1).drop('sector_index', axis=1).values[0]

                    mean_pattern = col_info.iloc[0,:].values
                    eigen1 = col_info.iloc[1,:].values
                    eigen2 = col_info.iloc[2,:].values
                    eigen1_true = [x/100 for x in eigen1]
                    eigen2_true = [x/100 for x in eigen2]
                    eigenpatterns = [eigen1_true, eigen2_true]

                    #histo, lines, votes = hough_transform_PIL(hits, layer_list, mean_pattern, eigenpatterns, layer_widths_col)
                    histo, lines, votes, hit_map = hit_array_HT(hits, full_hit_set, layer_list,
                                                            mean_pattern, eigenpatterns, layer_widths_col)
                    
                    #original = histo
                    #plt.imshow(histo)
                    #plt.colorbar()
                    
                    for i in range(iterations):
                        histo, votes, hit_map = update_votes(histo, lines, votes, hit_map, t1, t2)
                        #print('iteration = ', i)

                    #plt.imshow(histo)
                    #plt.colorbar()
                    #print('max = ' + str(histo.max()))
                    event_results.append([event_num, sector_index, histo, hit_map])
 
        time1 = time.time()
#        print('sector time =', time1-time0)

    #return(event_results)
    return(event_results)


def hit_array_HT(hits, full_hit_set, layer_list, mean, eigenpatterns, layer_widths):
    
    initial_VOTE = 1
    
    lines = []
    votes = []
    array_list = []
    hit_map = {}
    
    eigen1 = eigenpatterns[0]
    eigen2 = eigenpatterns[1]
     
    print(eigen1)
    print(eigen2)

    for layer in range(8):
        
        lines_i = []
        votes_i = []
        vote_array = np.zeros(shape = (size, size), dtype = np.uint8)
        #vote_array = np.full((size,size), [])
        width = layer_widths[layer] + 1
    
        ## 'indices' tells which hits to extract from hit list for given layer
    
        indices = [i for i, e in enumerate(layer_list) if e == layer]
                
        for hit_i in indices:
            
            if (eigen1[layer] == 0.0) and (eigen2[layer] == 0.0):
                eigen1[layer] = 0.00001
                eigen2[layer] = 0.00001

            if eigen2[layer] == 0.0:
                eigen2[layer] = 0.00001
 

            x1 = 0
            x2 = size
            y1 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] + (x2/2)*eigen1[layer]) + (x2/2))
            y2 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] - (x2/2)*eigen1[layer]) + (x2/2))
            
            LINE = np.array(get_bresenham_line(x1, y1, x2, y2, width))
            
            if len(LINE[0]) > 0:
                lines_i.append(LINE)
                line_votes = np.ones(len(LINE[0]))
                #line_votes = np.full(LINE.shape, full_hit_set[hit_i])
                votes_i.append(line_votes)
                bins = [LINE[:,i] for i in range(len(LINE[0]))]
                hit_map[str(full_hit_set[hit_i])] = bins
                #hit_map[LINE] = full_hit_set[hit_i]
            
            
            
        for l in lines_i:
            
            ## do we still need the vote_array to be updated by initial_VOTE? this prevents hits from the same layer 
            ## being added to the same bin
            
            vote_array[l[0],l[1]] = initial_VOTE
            #vote_array[l[0],l[1]].append()
        
        array_list.append(vote_array)
        lines.append(lines_i)
        votes.append(votes_i)
            
    histo = np.zeros(shape = vote_array.shape, dtype = np.float)
    histo = sum(array_list)
    #filtered = box_filter(histo, nn)
    
   
    return(histo, lines, np.array(votes), hit_map)
    #return(filtered, np.array(lines), np.array(votes), hit_map)


def update_votes(hough_image, lines, votes, hit_map, t1, t2):
    
    array_list = []
    keys = list(hit_map.keys())
    i=0 ## separate iterator for hit_map
    
    for layer in range(8):
        
        vote_array = np.zeros(shape = (size, size), dtype = np.float)
        
        lines_i = lines[layer]
        votes_i = votes[layer]
            
        for L in range(len(lines_i)):
                        
            line = lines_i[L]
            line_votes = votes_i[L]
            #hit_contrib = hit_map[keys[i]]

            pixels = hough_image[line[0], line[1]] - line_votes      
            
            if max(pixels) < t1:
                new_votes = 0*pixels
            else:    
                new_votes = pixels/pixels.max()
                 
            new_votes[new_votes < t2] = 0   
            
            ## update hit_map
            new_contrib = []
            for bin_i in range(len(new_votes)):
                if new_votes[bin_i] != 0:
                    new_contrib.append(line[:,bin_i])
            
            hit_map[keys[i]] = new_contrib
            i+=1

            vote_array[line[0],line[1]] = new_votes
            
            votes[layer][L] = new_votes
          
        #filtered_array = box_filter(vote_array, nn)
        #filtered_array, pooled_hit_map = box_filter(vote_array, hit_map, nn)
        #array_list.append(filtered_array)
        #array_list.append(vote_array)
        #updated_hit_map = hit_map_pooling(vote_array, filtered_array, hit_map)
    
        array_list.append(vote_array)
        
    total_image = sum(array_list)
    filtered_array = box_filter(total_image, nn)
    updated_hit_map = hit_map_pooling(total_image, filtered_array, hit_map)
    
    #new_hough_image = sum(array_list)
    #return(new_hough_image, votes, updated_hit_map)
    return(filtered_array, votes, updated_hit_map)



def hit_map_pooling(image, pooled, hit_map):
    
    """Apply max-pooling operation to the hit_map. The function takes
    the difference between the original and pooled image to find bins
    affected by the max-pooling. The maximum nearest neighbour is then
    located for each bin, and any hits contributing the the neighbour have
    the affected bin added to their hit_map"""
    
    diff = pooled - image
    non_zeros = np.array(np.where(diff!=0))
    coords = [list(non_zeros[:,i]) for i in range(len(non_zeros[0]))]
    
    #plt.imshow(diff)
    #plt.colorbar()
    
    #print('number of bins to update =', len(coords))
    
    for xy in coords:
        
        x = xy[0]
        y = xy[1]
        
        if (x <= (size-(nn+1))) and (y <= (size-(nn+1))):
        
            window_x = np.array([[x-nn, x, x+nn],[x-nn, x, x+nn],[x-nn, x, x+nn]])
            window_y = np.array([[y-nn, y-nn ,y-nn], [y, y, y], [y+nn, y+nn, y+nn]])
        
            maxpool = image[window_x, window_y]
            pool_coords = np.where(maxpool==maxpool.max())
            diff_coords = np.array([window_x[pool_coords[0],pool_coords[1]] , window_y[pool_coords[0],pool_coords[1]]])
            diff_coords_list = [list(diff_coords[:,i]) for i in range(len(diff_coords[0]))]

            for bin_i in diff_coords_list:
            
                keys = hit_map.keys()
                for key in keys:
                    bin_map = hit_map[key]
                    bin_map_list = []
                    for bin_j in bin_map:
                        bin_map_list.append(list(bin_j))
                    if bin_i in bin_map_list:
                        hit_map[key].append(xy)
        
    return(hit_map)


def check_muons(event_num):

    print(event_num)
    detector_elements = []
    sectors = []
    found_sectors = []
    coll_indicies = []
    
    event = muon_clusters[event_num]
    collections = event.colls
    
    ## Find all fired detector_elements in the event
    
    for coll_i in range(len(collections)):
        coll = collections[coll_i]
        hashId = coll.hashId
        detector_elements.append([coll_i, hashId])
    
    de_df = pd.DataFrame(detector_elements, columns=['coll_i', 'hashId'])
    #print(de_df)
    
    ## Loop through popular sectors and store the sector/sectorId if
    ## at least 7 of the hashIds contain hits
    
    for sector_index in popular_sectors:
        colls = []
        full_sector = dict_df[dict_df.sector_index == (sector_index)].values[0][:]
        sector = full_sector[1:]
        #if full_sector[0] in matched_sectors:
        if -1 in sector:
            pass
        else:
            counter = 0
            for hashId in sector:
                if hashId in de_df.hashId.values:
                    coll_index = de_df[de_df.hashId == hashId].values[0][0]    
                    colls.append(coll_index)
                    counter += 1
            if counter >= 7:
                found_sectors.append(sector_index)
                sectors.append(sector)
                coll_indicies.append(colls)
                    
    print(found_sectors)
    
    for num in range(len(found_sectors)):

        
        time0 = time.time()

        print('Sector', num+1, 'of', len(found_sectors))

        hits = []
        columns = []
        hashIds = []
        muons = []
        hit_set = []

        sector_index = found_sectors[num]
        
        if sector_index != cs:
            continue
        
        print('sector index', sector_index, flush=True)
        colls = coll_indicies[num]
        layer_widths_full = dc_info[dc_info.sector_index == sector_index]
        
        for coll_i in colls:
            coll = collections[coll_i]
            if coll.hashId in sectors[num]:
                for hit in coll.hits:

                    hit_set.append(hit)
                    hit_info = decode_hit(hit[0], maxStripP)
                    hits.append(hit_info[0])
                    hashIds.append(coll.hashId)
                    if hit_info[1] != -1:
                        columns.append(hit_info[1])

        ## need the condition here that there is at least one hit in 7 different layers, otherwise no point proceeding.
        ## allocate each hit to a layer

        print('number of hits', len(hits))
        
        full_hit_set = []
        layer_list = []
        for i in range(len(hashIds)):
            for layer_num in range(len(layer_element_list)):
                if hashIds[i] in layer_element_list[layer_num]:
                    layer_list.append(layer_num)
                    full_hit_set.append([layer_num, hit_set[i]])
                    
    return(full_hit_set)


def box_filter(array, nn):
    new_vote_array = nd.generic_filter(array, max, size=nn+2)
    return(new_vote_array)

## Process Events ##
####################

pile_up_results = event_HT_tracking(event_num)
pickle.dump(pile_up_results, open( "pile_up_results_event_" + str(event_num) + ".p", "wb" ))
#pickle.dump(pile_up_results, open( "muons_results_event_" + str(event_num) + ".p", "wb" ))








