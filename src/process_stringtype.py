import math
import nltk
import re
import numpy as np
import pandas as pd
from functools import partial
from geopy.geocoders import Nominatim
from word2number import w2n


def calculate_xyz(ll):
    """ Convert (GPS) coordinates to (X, Y, Z) coordinates. Inspired by Nat Wilson's post at
    https://gis.stackexchange.com/questions/212723/how-can-i-convert-lon-lat-coordinates-to-x-y

    :param ll: a List of Floats representing latitude/longitude values.
    :return: a List of Floats ([x-coordinate, y-coordinate, z-coordinate]) of each latitude/longitude value.
    """
    x = math.sin(math.pi / 2 - ll[0]) * math.cos(ll[1])
    y = math.sin(math.pi / 2 - ll[0]) * math.sin(ll[1])
    z = math.cos(math.pi / 2 - ll[0])
    return [x, y, z]


def process_coordinate(df, orig_df):
    """ Process coordinates into multiple useful features.

    :param df: a pandas DataFrame consisting of Strings that represent coordinate data (unique values only).
    :param orig_df: a pandas DataFrame consisting of the original coordinate column.
    :return: a pandas DataFrame consisting of latlong, zipcodes, country_codes, and xyz-coordinates;
             a (List of) Integer(s) indicating the encoding type(s) required for each column.
    """
    # Set up additional columns that will be returned alongside the transformed coordinates
    latlong = list(df.values.flatten())
    changes, changes_country, changes_xyz, changes_zipcode = {}, {}, {}, {}

    # To check whether we are dealing with coordinate pairs or not
    single_coord = True

    # Instantiate geopy session to retrieve useful data
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    for i in range(len(latlong)):
        # Separate values into parts that are easier to work with
        val = re.split('[NSZEOW]', latlong[i])
        val = val[1:]
        val = [val[j].split('.') for j in range(len(val))]

        # We have to process coordinates with just north/south or east/west differently
        if len(val) == 2:
            val = [val[0] + val[1]]
        elif len(val) == 4:
            val = [val[0] + val[1], val[2] + val[3]]

        # Convert coordinates into latitude/longitude values, inspired by
        # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
        for j in range(len(val)):
            if val[j][0] in 'SZW':
                val[j] = -(int(val[j][1]) + int(val[j][2]) / 60 + int(val[j][3]) / 3600)
            else:
                val[j] = int(val[j][1]) + int(val[j][2]) / 60 + int(val[j][3]) / 3600

        if len(val) == 2:
            single_coord = False
            # Convert GPS coordinates to XYZ coordinates
            xyz = calculate_xyz(val)

            # Convert coordinates to additional data using geopy
            latlong_info = reverse('{}, {}'.format(val[0], val[1]))

            # It can be the case that the latlong values point to a deserted place. In this case, there is no zipcode
            # or country_code and we mark those places as 'unknown'
            if latlong_info:
                if 'postcode' in latlong_info.raw['address']:
                    zipcode = latlong_info.raw['address']['postcode']
                else:
                    zipcode = 'unknown'
                if 'country_code' in latlong_info.raw['address']:
                    country = latlong_info.raw['address']['country_code']
                else:
                    country = 'unknown'
            else:
                zipcode = 'unknown'
                country = 'unknown'
            # Insert the obtained values in the corresponding dicts
            changes_country[latlong[i]] = country
            changes_xyz[latlong[i]] = xyz
            changes_zipcode[latlong[i]] = zipcode

        # Insert obtained latlong into the corresponding dict
        changes[latlong[i]] = val

    if not single_coord:
        # In case we are dealing with coordinate pairs, add the additional info
        orig_df[str(df.columns[0]) + '_country'] = orig_df[df.columns[0]].map(changes_country)
        orig_df[str(df.columns[0]) + '_zipcode'] = orig_df[df.columns[0]].map(changes_zipcode)
        orig_df[str(df.columns[0]) + '_xyz'] = orig_df[df.columns[0]].map(changes_xyz)

    # Replace the original values for the encoded ones
    orig_df = orig_df.replace({df.columns[0]: {df.columns[0]: changes}})

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    if single_coord:
        return orig_df, 2
    else:
        return orig_df, [2, 1, 1, 2]


def process_day(df):
    """ Process days into easy-to-encode strings. We assume that Strings that represent days are always at the front.

    :param df: a pandas DataFrame consisting of Strings that represent days.
    :return: a pandas DataFrame consisting of Strings that represent abbreviated days;
             an Integer indicating the encoding type required.
    """
    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return df[df.columns[0]].str[:2], 1


def process_email(df):
    """ Process e-mails into easy-to-encode strings with (possibly) enhanced properties.

    :param df: a pandas DataFrame consisting of Strings that represent e-mails.
    :return: a Dictionary that maps the old values to the new values (String -> String);
             an Integer representing the encoding type required.
    """
    emails = list(df.values.flatten())
    changes = {}

    for i in range(len(emails)):
        # Split between name and domain of e-mail + remove top-level domains
        val = emails[i].split("@", 1)
        val[1] = val[1].split(".", 1)[0]
        val = " ".join(val)
        changes[emails[i]] = val

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 1


def process_filepath_url(df, type):
    """ Process filepaths and URLs into easy-to-encode strings with (possibly) enhanced properties.

    :param df: a pandas DataFrame consisting of Strings that represent filepaths or URLs.
    :param type: a String representing whether the column contains 'filepath' or 'url' entries.
    :return: a pandas DataFrame consisting of easy-to-encode filepath or URL Strings;
             an Integer representing the encoding type required.
    """
    data = list(df.values.flatten())
    changes = {}

    for i in range(len(data)):
        val = data[i]
        if type == 'url':
            if '://' in data[i]:
                val = val.split("://", 1)[1]
        val = re.split("[^a-zA-Z0-9]", val)
        val = list(filter(lambda a: a != "", val))
        val = " ".join(val)
        changes[data[i]] = val

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 1


def process_month(df):
    """ Process months into easy-to-encode strings. We assume that Strings that represent months are always at the
    front.

    :param df: a pandas DataFrame consisting of Strings that represent months.
    :return: a Dictionary that maps the old values to the new values (String -> String);
             an Integer representing the encoding type required.
    """
    months = list(df.values.flatten())
    changes = {}

    # Mapping from months to numerical / encoded values
    month_to_int = {
        'jan': ['01', '000000000001'],
        'feb': ['02', '000000000010'],
        'mar': ['03', '000000000100'],
        'apr': ['04', '000000001000'],
        'may': ['05', '000000010000'],
        'jun': ['06', '000000100000'],
        'jul': ['07', '000001000000'],
        'aug': ['08', '000010000000'],
        'sep': ['09', '000100000000'],
        'oct': ['10', '001000000000'],
        'nov': ['11', '010000000000'],
        'dec': ['12', '100000000000']
    }

    for i in range(len(months)):
        # Set the dd/mmm/yy variables and split the data based on characters in each value
        dd, mmm, yy = '', '', ''
        split_vals = re.split('[,.\-_ ]', months[i])

        if len(split_vals) == 1:
            # Having only one value in split_vals implies we are only dealing with months
            # (because of how the pfsm works).
            changes[months[i]] = month_to_int[months[i][:3].lower()][1]
        elif len(split_vals) == 2:
            # Having two values in split_vals almost always implies that the data contains month + year
            for val in split_vals:
                if val[0] == '\'':
                    yy = val[1:]
                elif val.isdigit():
                    yy = val[2:]
                else:
                    mmm = val[:3]
            changes[months[i]] = int(yy + month_to_int[mmm.lower()][0] + '01')
        else:
            # Having >2 values in split_vals means that we are dealing with day + month + year
            for val in split_vals:
                print(val)
                if val[0] == '\'':
                    yy = val[1:]
                elif val.isdigit():
                    if len(val) == 4:
                        yy = val[2:]
                    # TODO: improve this int(val) > 28 constraint
                    elif int(val) > 28 or dd != '':
                        yy = val
                    else:
                        dd = val
                else:
                    mmm = val[:3]
            changes[months[i]] = int(yy + month_to_int[mmm.lower()][0] + dd)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 2


def process_numerical(df):
    """ Convert numerical strings (such as ranges, units, etc.) into ordinally-encoded data.

    :param df: a pandas DataFrame consisting of Strings that represent numerical values.
    :return: a Dictionary that maps the old values to the new values (String -> Integer);
             an Integer representing the encoding type required.
    """
    nums = list(df.values.flatten())
    processed_nums = []

    for i in range(len(nums)):
        if re.fullmatch('[0-9]+ ?[-:_/(to)] ?[0-9]+', nums[i]):
            # Take the mean of the range
            digits = np.mean([int(s) for s in re.split(' ?[-:_/(to)] ?', nums[i]) if s.isdigit()])
            processed_nums.append([nums[i], digits])
        else:
            # only take the relevant numbers
            digits = int(''.join([s for s in nums[i] if s.isdigit()]))
            processed_nums.append([nums[i], digits])

    # Sort the data based on the processed numbers
    processed_nums.sort(key=lambda x: x[1])

    # Map the original values to the (ordinal encoded) values
    changes = {processed_nums[i][0]: i for i in range(len(processed_nums))}

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 2


def process_ordinal(df):
    """ Process Strings that represent rank into ordinal-encoded numbers, adhering to the rank.

    :param df: a pandas DataFrame consisting of Strings that represent ordered numbers.
    :return: a Dictionary that maps the old values to the new values (String -> Integer);
             an Integer representing the encoding type required.
    """
    ordinals = list(df.values.flatten())
    original_ord = ordinals.copy()
    changes = {}

    # Set up a small dictionary of ordered words that are irregular
    irreg_ord_dict = {
        'first': 'one',
        'second': 'two',
        'third': 'three',
        'fifth': 'five',
        'eighth': 'eight',
        'ninth': 'nine',
        'twelfth': 'twelve',
    }

    for i in range(len(ordinals)):
        # Check if a value corresponds to one of the irregular words
        for key, item in irreg_ord_dict.items():
            if key in ordinals[i].lower():
                ordinals[i] = ordinals[i].lower().replace(key, " " + item)
                break

        # Remove the suffix that gives each word their order and replace some letters if necessary
        if ordinals[i][-2:] == 'th':
            ordinals[i] = ordinals[i][:-2]
            if ordinals[i][-2:] == 'ie':
                ordinals[i] = ordinals[i][:-2] + 'y'

        # Use word2number to translate (spelled-out) string numbers directly to integer numbers
        ordinals[i] = w2n.word_to_num(ordinals[i])
        changes[original_ord[i]] = ordinals[i]

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 2


# TODO: add sentence/word embeddings to it as well
def process_sentence(df):
    """ Process sentences into a list of nouns for easier encoding.

    :param df: a pandas DataFrame consisting of Strings that represent sentences.
    :return: a Dictionary that maps the old values to the new values (String -> String);
             an Integer representing the encoding type required.
    """
    sentences = list(df.values.flatten())
    changes = {}

    for sent in sentences:
        # Using straight-up nltk word tokenizer
        tokenized = nltk.word_tokenize(sent)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos == 'NN']
        changes[sent] = " ".join(nouns)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 1


def process_zipcode(df, orig_df):
    """ Process coordinates into multiple useful features.

    :param df: a pandas DataFrame consisting of Strings that represent zipcodes.
    :param orig_df: a pandas DataFrame consisting of Strings that represent the original zipcode column.
    :return: a pandas DataFrame consisting of zipcodes, latlong, country_codes, and xyz-coordinates;
             a (List of) Integers(s) indicating the encoding type(s) required for each column.
    """
    # Instantiate geopy session
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    # Flatten dataframe to list + create new dictionary for each feature
    zipcodes = list(df.values.flatten())
    changes, changes_city, changes_country, changes_latlong, changes_xyz = {}, {}, {}, {}, {}

    for i in range(len(zipcodes)):
        # Remove any spaces (this is basically all the pre-processing we need for just zipcodes)
        val = zipcodes[i].replace(' ', '')
        zip_info = geolocator.geocode(val)
        latlong = [zip_info.latitude, zip_info.longitude]

        # Convert latitude/longitude to xyz
        xyz = calculate_xyz([zip_info.latitude, zip_info.longitude])

        # Gather more information using geopy
        latlong_info = reverse('{}, {}'.format(zip_info.latitude, zip_info.longitude)).raw
        country = latlong_info['address']['country_code']
        if 'city' in latlong_info['address']:
            city = latlong_info['address']['city']
        else:
            city = 'unknown'

        # Insert the obtained values in the corresponding dicts
        changes_city[zipcodes[i]] = city
        changes_country[zipcodes[i]] = country
        changes_latlong[zipcodes[i]] = latlong
        changes_xyz[zipcodes[i]] = xyz
        changes[zipcodes[i]] = val

    # Map the updated + new values to the corresponding values in the original DataFrame
    orig_df = orig_df.replace({df.columns[0]: changes})
    orig_df[str(df.columns[0]) + '_city'] = orig_df[df.columns[0]].map(changes_city)
    orig_df[str(df.columns[0]) + '_country'] = orig_df[df.columns[0]].map(changes_country)
    orig_df[str(df.columns[0]) + '_latlong'] = orig_df[df.columns[0]].map(changes_latlong)
    orig_df[str(df.columns[0]) + '_xyz'] = orig_df[df.columns[0]].map(changes_xyz)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return orig_df, [1, 1, 2, 1]


def run(df, stringtype):
    """ Handle the processing of data in a unique string column into easy-to-encode (+ enhanced) data.

    :param df: a pandas DataFrame consisting of unique String entries.
    :param stringtype: a String indicating which unique String type the column df consists of.
    :return: a pandas DataFrame consisting of column(s) containing processed (+ additional) data; a (List of) Integer(s)
             indicating the encoding type(s) required for each column.
    """
    if stringtype in ['day']:
        return eval('process_{}(df)'.format(stringtype))
    else:
        df_unique = pd.DataFrame(df[df.columns[0]].unique(), columns=[df.columns[0]])
        if stringtype in ['filepath', 'url']:
            result, encode = process_filepath_url(df_unique, stringtype)
        elif stringtype in ['coordinate', 'zipcode']:
            result, encode = eval('process_{}(df_unique, df)'.format(stringtype))
            return result, encode
        else:
            result, encode = eval('process_{}(df_unique)'.format(stringtype))
        return df.replace({df.columns[0]: result}), encode

# data = pd.DataFrame(['<100', 'over 600', 'over 600', '100-300', '300-600'])
