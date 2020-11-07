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

clusters = read_clusters(data_dir + 'hit_events_pileup_200_eta0.1-0.3.bin')
muon_clusters = read_clusters(data_dir + 'hit_events_no_pileup_eta0.1-0.3.bin')

complete_sector_ids = dict_df.loc[dict_df.isin([-1]).sum(1) == 0].sector_index.values
popular_sectors = []
for sector in complete_sector_ids:
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


def hough_transform_PIL(hits, layer_list, mean, eigenpatterns, layer_widths):
    
    initial_VOTE = 1
    
    lines = []
    votes = []
    array_list = []
    
    eigen1 = eigenpatterns[0]
    eigen2 = eigenpatterns[1]
        
    for layer in range(8):
        
        lines_i = []
        votes_i = []
        vote_array = np.zeros(shape = (size, size), dtype = np.uint8)
        width = layer_widths[layer] + 1
        indices = [i for i, e in enumerate(layer_list) if e == layer]
                
        for hit_i in indices:
            
            x1 = 0
            x2 = size
            y1 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] + (x2/2)*eigen1[layer]) + (x2/2))
            y2 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] - (x2/2)*eigen1[layer]) + (x2/2))
            
            LINE = get_bresenham_line(x1, y1, x2, y2, width)
            
            if len(LINE[0]) > 0:
                lines_i.append(LINE)
                line_votes = np.ones(len(LINE[0]))
                votes_i.append(line_votes)
            
        for l in lines_i:
            
            vote_array[l[0],l[1]] = initial_VOTE
        
        array_list.append(vote_array)
        lines.append(lines_i)
        votes.append(votes_i)
            
    histo = np.zeros(shape = vote_array.shape, dtype = np.float)
    histo = sum(array_list)
        
    return(histo, np.array(lines), np.array(votes))



def update_votes(hough_image, lines, votes, t1, t2):
    
    array_list = []
    
    for layer in range(8):
        
        t0 = time.time()
        vote_array = np.zeros(shape = (size, size), dtype = np.float)
        
        lines_i = lines[layer]
        votes_i = votes[layer]
            
        for L in range(len(lines_i)):
                        
            line = lines_i[L]
            line_votes = votes_i[L]

            pixels = hough_image[line[0], line[1]] - line_votes      
            
            if max(pixels) < t1:
                new_votes = 0*pixels
            else:    
                new_votes = pixels/pixels.max()
                
            new_votes[new_votes < t2] = 0   

            vote_array[line[0],line[1]] = new_votes
            
            votes[layer][L] = new_votes
        
        filtered_array = box_filter(vote_array, nn)
        array_list.append(filtered_array)
    
        
    new_hough_image = sum(array_list)
    return(new_hough_image, votes)


def old_box_filter(binary_image, nn):

    hough_image = np.zeros(shape = (size, size), dtype = np.uint8)
    new_binary_image = np.zeros(shape = (size, size), dtype = np.uint8)

    if nn == 0:
        for row in range(len(binary_image)):
            for column in range(len(binary_image[0])):
                pixel = binary_image[row][column]
                layer_count = bin(pixel).count("1")
                hough_image[row][column] = layer_count
    
    for i in range(len(binary_image)-2):   
        row = i + 1
        for j in range(len(binary_image[i])-2):
            column = j + 1
                
            window = []
            rows = binary_image[i-nn:i+nn+1]
        
            for r in rows:
                window.append(list((r[j-nn:j+nn+1])))
        
            window_array = np.array(window, dtype=np.uint8)
        
            accumulated_vote = np.bitwise_or.reduce(window_array.flatten())
            layer_count = bin(accumulated_vote).count("1")
        
            new_binary_image[row][column] = accumulated_vote
            hough_image[row][column] = layer_count
        
    return(new_binary_image, hough_image)


def make_binary_image(hits, layer_list, mean, eigenpatterns, layer_widths):
    
    lines = []
    array_list = []
    
    initial_votes = []
    for l in range(8):
        initial_votes.append(2**l)
    
    eigen1 = eigenpatterns[0]
    eigen2 = eigenpatterns[1]
    
    for layer in range(8):
        
        lines_i = []
        vote_array = np.zeros(shape = (size, size), dtype = np.uint8)
        width = layer_widths[layer] + 1
        indices = [i for i, e in enumerate(layer_list) if e == layer]
        
        for hit_i in indices:
            
            x1 = 0
            x2 = size
            y1 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] + (x2/2)*eigen1[layer]) + (x2/2))
            y2 = round((1 / eigen2[layer])*(hits[hit_i] - mean[layer] - (x2/2)*eigen1[layer]) + (x2/2))
            
            lines_i.append(get_bresenham_line(x1, y1, x2, y2, width))
            lines.append(get_bresenham_line(x1, y1, x2, y2, width))
                
        for l in lines_i:
            
            vote_array[l[0],l[1]] = initial_votes[layer]
        
        array_list.append(vote_array)
            
    histo = np.zeros(shape = vote_array.shape, dtype = np.uint8)
    histo = sum(array_list)
        
    return(histo, lines)




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
              
    
    ## Consider each found_sector for tracking
        
    event_results = []
                
    for num in range(len(found_sectors)):
            
        #time0 = time.time()
        
        #print('Sector', num+1, 'of', len(found_sectors))
            
        hits = []
        columns = []
        hashIds = []
    
        sector_index = found_sectors[num]
        print('sector index', sector_index, flush=True)
        colls = coll_indicies[num]
        layer_widths_full = dc_info[dc_info.sector_index == sector_index]

        for coll_i in colls:
            coll = collections[coll_i]
            if coll.hashId in sectors[num]:
                for hit in coll.hits:
                    
                    hit_info = decode_hit(hit[0], maxStripP)
                    hits.append(hit_info[0])
                    hashIds.append(coll.hashId)
                    if hit_info[1] != -1:
                        columns.append(hit_info[1])
            
        ## need the condition here that there is at least one hit in 7 different layers, otherwise no point proceeding.  
        ## allocate each hit to a layer
        
        layer_list = []
        for i in range(len(hashIds)):
            for layer_num in range(len(layer_element_list)):
                if hashIds[i] in layer_element_list[layer_num]:
                    layer_list.append(layer_num)
    
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
            
                histo, lines, votes = hough_transform_PIL(hits, layer_list, mean_pattern, eigenpatterns, layer_widths_col)

                #plt.imshow(histo)
                #plt.colorbar()
                
                for i in range(iterations):
                    histo, votes = update_votes(histo, lines, votes, t1, t2)
                        #print('iteration = ', i)
                
                #plt.imshow(histo)
                #plt.colorbar()
                #print('max = ' + str(histo.max()))
                
                event_results.append([event_num, sector_index, histo])
             
        #time1 = time.time()
        #print('sector time =', time1-time0)
             
    return(event_results)



def generate_muon_images(event_num):
    
    """
    Processes a range of full events to produce an output Hough image with
    localised track recognition.
    
    """
    
    muon_images = []
            
    print('Event', event_num)
    
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
            if counter >= 6:
                found_sectors.append(sector_index)
                sectors.append(sector)
                coll_indicies.append(colls)
              
    
    ## Consider each found_sector for tracking
         
    for num in range(len(found_sectors)):
        
        print('Sector', num+1, 'of', len(found_sectors), flush=True)
            
        hits = []
        columns = []
        hashIds = []
    
        sector_index = found_sectors[num]
        print('sector index', sector_index)
        colls = coll_indicies[num]
        layer_widths_full = dc_info[dc_info.sector_index == sector_index]

        for coll_i in colls:
            coll = collections[coll_i]
            if coll.hashId in sectors[num]:
                for hit in coll.hits:
                    
                    hit_info = decode_hit(hit[0], maxStripP)
                    hits.append(hit_info[0])
                    hashIds.append(coll.hashId)
                    if hit_info[1] != -1:
                        columns.append(hit_info[1])
            
        ## need the condition here that there is at least one hit in 7 different layers, otherwise no point proceeding.  
        ## allocate each hit to a layer
        
        layer_list = []
        for i in range(len(hashIds)):
            for layer_num in range(len(layer_element_list)):
                if hashIds[i] in layer_element_list[layer_num]:
                    layer_list.append(layer_num)
    
        #if len(hits) == 7:
        #        correct_layers(hits, layer_list)
                
        print('number of hits =', len(hits))
        
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
            
                binary_image = make_binary_image(hits, layer_list, mean_pattern, eigenpatterns, layer_widths_col)[0]
                #plt.imshow(binary_image)
                #plt.colorbar()
                muon_image = old_box_filter(binary_image, 2)[1]
                muon_images.append([event_num, sector_index, muon_image])
        

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
            
                    binary_image = make_binary_image(hits, layer_list, mean_pattern, eigenpatterns, layer_widths_col)[0]
                    #plt.imshow(binary_image)
                    #plt.colorbar()
                    muon_image = old_box_filter(binary_image, 2)[1]
                    muon_images.append([event_num, sector_index, muon_image]) 
            
    return(muon_images)



####################
## Process Events ##
####################

#pool = ThreadPool(20)

#pile_up_results = pool.map(event_HT_tracking, event_nums)
#muon_results = pool.map(generate_muon_images, event_nums)

#efficiency_results = process_results(pile_up_results, muon_results, t_match)

#print('Efficiency =', efficiency_results[0])
#print('Number of fake tracks =', efficiency_results[1])

#print('Events processed - storing results...', flush=True)


def analyse_sectors(t_match, results):
    
    tracks = []

    for sector in range(len(results)):
        image = results[sector][2]
        sector_index = results[sector][1]
        print('Sector', sector_index, 'being analysed', flush=True)
        blob_results = find_dog_peaks(image)
        peak_values = blob_results[0]
        blobs = blob_results[1]
        
        for value_i in range(len(peak_values)):
            value = peak_values[value_i]
            blob = blobs[value_i]
            if value >= t_match:
                tracks.append([sector_index, value, image])
                
    return(tracks)

def process_results(pile_up_results, muon_results, t_match, event_num):
    
    matched_tracks = []
    event_efficiency = 0
 
    #pile_up_images = pile_up_results[0]
    pile_up_tracks = analyse_sectors(t_match, pile_up_results)

    muon_df = pd.DataFrame(muon_results, columns=['event_num', 'sectorIdx', 'image'])
    
    print('Comparing tracks with muon images...', flush=True)

    for track in pile_up_tracks:
        sector = track[0]
        if sector in muon_df.values[:,1]:
            muon_image = muon_df[muon_df.sectorIdx == sector].values[0][2]
            image = track[2]
            coords = np.where(image==image.max())
            muon_count = muon_image[coords]
            if any(x > 6 for x in muon_count):
                matched_tracks.append(track)
                
    if len(matched_tracks) >= 1:
        event_efficiency = 1
    fake_tracks = len(pile_up_tracks) - len(matched_tracks)
    return(event_num, event_efficiency, fake_tracks, pile_up_tracks, matched_tracks)



pile_up_results = event_HT_tracking(event_num)
muon_results = generate_muon_images(event_num)
efficiencies = process_results(pile_up_results, muon_results, t_match, event_num)

print("====================")
print("EVENT NUMBER " + str(event_num) + ":")
print("====================")
print('')
if efficiencies[1] == 1:
    print("Muon found!")
else:
    print("No Muon found!")




pickle.dump(pile_up_results, open( "pile_up_results_event_" + str(event_num) + ".p", "wb" ))
pickle.dump(muon_results, open( "muon_results_event_" + str(event_num) + ".p", "wb" ))

efficiencies = process_results(pile_up_results, muon_results, t_match, event_num)
pickle.dump(efficiencies, open( "efficiency_event_" + str(event_num) + ".p", "wb" ))










