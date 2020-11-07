import numpy as np
import pandas as pd 
import scipy.ndimage as nd
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from math import sqrt

def event_HT_tracking(event_num):
    
    """
    Processes a range of full events to produce an output Hough image with
    localised track recognition.
    
    """
    results = []
        
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
    
        if len(hits) == 7:
                correct_layers(hits, layer_list)
        
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
                
                event_results.append([sector_index, histo])
             
        #time1 = time.time()
        #print('sector time =', time1-time0)
            
    results.append(event_results)
            
    return(results, lines)

def generate_muon_image(event_num):
    
    """
    Processes a range of full events to produce an output Hough image with
    localised track recognition.
    
    """
    
    results = []
            
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
            if counter >= 7:
                found_sectors.append(sector_index)
                sectors.append(sector)
                coll_indicies.append(colls)
              
    
    ## Consider each found_sector for tracking
        
    event_results = []
                
    for num in range(len(found_sectors)):
        
        print('Sector', num+1, 'of', len(found_sectors))
            
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
    
        if len(hits) == 7:
                correct_layers(hits, layer_list)
                
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
                results.append([sector_index, muon_image])
            
            
    return(results)


class CollectionClass(object) :
	def __init__(self, h, nH) :
		self.hashId = h
		self.nHits = nH
		self.hits = None


class EventClass(object) :
	def __init__(self, id, nC) :
		self.eventId = id
		self.nColls = nC
		self.colls = []


def read_clusters(filename):

# Create custom numpy data types

    dtype_list = []
    dtype_list.append(('hitData', 'uint16'))
    dtype_list.append(('subEventId', 'uint32'))
    dtype_list.append(('barCode', 'uint32'))

    hit_data_type = np.dtype(dtype_list)

    dtype_list = []
    dtype_list.append(('hashId', 'int32'))
    dtype_list.append(('nHits', 'uint32'))

    coll_data_type = np.dtype(dtype_list)

    dtype_list = []
    dtype_list.append(('eventId', 'int32'))
    dtype_list.append(('nColls', 'uint32'))

    event_data_type = np.dtype(dtype_list)

    dtype_list = []
    dtype_list.append(('nEvents', 'int32'))

    header_data_type = np.dtype(dtype_list)


    hitEvents = []

    with open(filename, "rb") as f :
        h = np.fromfile(f, dtype = header_data_type, count = 1)
        for i in range(h[0][0]) :
            e = np.fromfile(f, dtype = event_data_type, count = 1)
            #print(e)
            event = EventClass(e[0][0], e[0][1])
            for j in range(e[0][1]) :
                c = np.fromfile(f, dtype = coll_data_type, count = 1)
                coll = CollectionClass(c[0][0], c[0][1])
                coll.hits = np.fromfile(f, dtype = hit_data_type, count = c[0][1])
                event.colls.append(coll)
            hitEvents.append(event)

    return(hitEvents)


def correct_layers(hits, layer_list):
    
    x = layer_list[0]
    for i in range(len(layer_list) - 1):
        if x != i:
            hits.insert(i+1, -1)
            break
        x = layer_list[i + 1]

#def MAX(window):
#    return max(window)

def box_filter(array, nn):
    new_vote_array = nd.generic_filter(array, max, size=3)
    return(new_vote_array)

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
        width = layer_widths[layer]
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
        width = layer_widths[layer]
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



def find_dog_peaks(image):
        
    image_gray = rgb2gray(image)
    dog_blobs = blob_dog(image_gray, max_sigma=5, threshold=.1)
    dog_blobs[:, 2] = dog_blobs[:, 2] * sqrt(2)
    
    max_values = []
    peaks = []
    coords = []
    
    #print('Number of blobs detected =', len(dog_blobs))
    
    for i in range(len(dog_blobs)):
        
        f,ax = plt.subplots()
        blob = dog_blobs[i]
        coords.append(blob)
        ax.imshow(image)   
        y, x, r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
        ax.add_patch(c)
    
        #plt.show()

        x_range = [int(blob[0] + int(blob[2])), int(blob[0] - int(blob[2]))]
        y_range = [int(blob[1] + int(blob[2])), int(blob[1] - int(blob[2]))]
    
        x_slice = image[x_range[1]:x_range[0]]
    
        blob_list = []
        for y_slice in x_slice:
            column = y_slice[y_range[1]:y_range[0]]
            blob_list.append(column)
        blob_array = np.array(blob_list)
                
        if blob_array.size != 0:
            max_values.append(blob_array.max())
            #print('max_value = ', blob_array.max())
            peaks.append(blob_array)
            
    return(max_values, peaks, coords)


def analyse_sectors(t_match, results):
    
    tracks = []

    for sector in range(len(results)):
        image = results[sector][1]
        sector_index = results[sector][0]
        blob_results = find_dog_peaks(image)
        peak_values = blob_results[0]
        blobs = blob_results[1]
        
        for value_i in range(len(peak_values)):
            value = peak_values[value_i]
            blob = blobs[value_i]
            if value >= t_match:
                tracks.append([sector_index, value, image])
                
    return(tracks)
