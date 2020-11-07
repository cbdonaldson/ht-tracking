###########################################
###### DECODING / ENCODING FUCNTIONS ######
###########################################

## Information about each hit (strip/column positions, dc_bits) can be compressed and
## encoded in a single 8-bit number (np.uint8). The functions defined here will encode
## and decode the hit information, and isPixel will simply distinguish hits in pixel layers
## from strip layers.
##
## encodeHit:
##
## Requires input of (strip, column, dc_bits). For strip-layer hits, the column entry is
## '-1', and the encoded number represents 4*(strip postion) + dc_bits. For pixel layers,
## the column mesurement is multipled by 11 (width of pixel) to relfect the two-
## dimensional measurement, with dc_bits added. The addition of 128 places all pixel
## measurements within the range (128-255), distinct from the strip measurements (0-127).
##
## isPixel / decodeHit:
##
## Requires uint8 input. (h & 0x80) checks for h greater or less than 128, determining
## whether the hit is pixel or strip layer. The operator ((h & 0x7C) >> 2) splits the
## remaining 128 possible encoded values into 4 segments ranging from 0-31. Each segment
## corresponds to a number of dc_bits (0-3), chosen by (h & 0x03).
##

import numpy as np

def encode_strip(hit, maxStripP):

    h = np.uint16(0x0000)
    strip = np.uint16(hit[0])
    h += (strip << 2)

    return(h)

def encode_pixel(hit, maxStripP, col):

    h = np.uint16(0x8000)
    row = int(hit[0]) % maxStripP
    h += ((row + col*maxStripP) << 2)

    return(h)


def encode_hit(hit, maxStripP):

    if len(hit) == 1:
        h = np.uint16(0x0000)
        strip = hit[0]
#        dc_bits = hit[1]
        h += (strip << 2)
    else:
        h = np.uint16(0x8000)
        strip = hit[0]
        column = hit[1]
#        dc_bits = hit[2]
        h += ((strip + column*maxStripP) << 2)

    return(h)


def decode_hit(h, maxStripP):

    #if h.dtype != np.uint16:
    #    print('Wrong data type - uint16 required!')
    #    return(-1,-1,-1)

    if (h & 0x8000) != 0:
        dont_care_bits = h & 0x0003
        col_row = (h & 0x7FFC) >> 2
        strip = col_row % maxStripP
        column = col_row // maxStripP
        return(strip, column, dont_care_bits)

    else:
        dont_care_bits = h & 0x0003
        strip = (h & 0x7FFC) >> 2
        return (strip, -1, dont_care_bits)

def isPixel(h) :
        if h.dtype != np.uint16 :
                print('Wrong data type - uint16 required')
                return False

        return (h & 0x8000)!=0


def decode_sector(pattern_values, maxStripP):

    pattern_list = []

    for pIdx in range(len(pattern_values)) :
        single_pattern = []
        for layer in range(8):
            hit = decode_hit(pattern_values[pIdx][layer], maxStripP)
            if layer == 0 :
                single_pattern.append(np.float64(hit[0])) ## row
                single_pattern.append(np.float64(hit[1])) ## col
                #dc_bits = hit[2]
                #if dc_bits == 3:
                #    single_pattern.append(np.float(dc_bits - 1)) ## no option for 2
                #else:
                #    single_pattern.append(np.float(dc_bits))
            else:
                single_pattern.append(np.float64(hit[0])) ## strip only
                #dc_bits = hit[2]
                #if dc_bits == 3:
                #    single_pattern.append(np.float(dc_bits - 1))
                #else:
                #    single_pattern.append(np.float(dc_bits))
        pattern_list.append(single_pattern)

    return(pattern_list)



def encode_sector(pattern_values, maxStripP, col):
    
    encoded_patterns = []
    for pattern in np.around(pattern_values).astype(int):
        encoded_pattern = []
        for layer in range(8):
            if layer == 0:
                hit_info = pattern[:1]
                #if hit_info[2] == 2:
                #    hit_info[2] = 3
                h = encode_pixel(hit_info, maxStripP, col)
                encoded_pattern.append(h)
            else:
                hit_info = [pattern[layer]]
                #if hit_info[1] == 2:
                #    hit_info[1] = 3
                h = encode_strip(hit_info, maxStripP)
                encoded_pattern.append(h)
        encoded_patterns.append(encoded_pattern)

    return(encoded_patterns)



def encode_wc_sector_no_pixel(pattern_values, maxStripP, num_wildcards):
    
    encoded_patterns = []
    for pattern in np.around(pattern_values).astype(int):
        encoded_pattern = []
        for layer in range(8 - num_wildcards):
            hit_info = [pattern[layer]]
            h = encode_strip(hit_info, maxStripP)
            encoded_pattern.append(h)
        encoded_patterns.append(encoded_pattern)

    return(encoded_patterns)



def encode_wc_sector(pattern_values, maxStripP, col, num_wildcards):
    
    encoded_patterns = []
    for pattern in np.around(pattern_values).astype(int):
        encoded_pattern = []
        for layer in range(8 - num_wildcards):
            if layer == 0:
                hit_info = pattern[:1]
                h = encode_pixel(hit_info, maxStripP, col)
                encoded_pattern.append(h)
            else:
                hit_info = [pattern[layer]]
                h = encode_strip(hit_info, maxStripP)
                encoded_pattern.append(h)
        encoded_patterns.append(encoded_pattern)

    return(encoded_patterns)
