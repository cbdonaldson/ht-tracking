import sys
import numpy as np
import pandas as pd
import time
from math import sqrt
import pickle
from tracking_functions import find_dog_peaks
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

t0 = time.time()

data_dir = sys.argv[1]
t_match = float(sys.argv[2])


###########
## SETUP ##
###########

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


def process_results(pile_up_results, muon_results, t_match):
    
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
    return(event_efficiency, fake_tracks, pile_up_tracks, matched_tracks)





pile_up_results = pickle.load(open(data_dir + "pile_up_results.p", "rb"))
muon_results = pickle.load(open(data_dir +"muon_results.p", "rb"))


##############
## Analysis ##
##############


efficiency_results = process_results(pile_up_results, muon_results, t_match)

#print('Efficiency =', efficiency_results[0])
#print('Fake tracks =', efficiency_results[1])

pickle.dump(efficiency_results, open("efficiency_results.p", "wb"))






