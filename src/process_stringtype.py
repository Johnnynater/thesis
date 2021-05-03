import re
import math
import nltk
import pandas as pd
from functools import partial
from geopy.geocoders import Nominatim
from word2number import w2n


def calculate_xyz(ll):
    """ Convert (GPS) coordinates to (X, Y, Z) coordinates. Inspired by Nat Wilson's post at
    https://gis.stackexchange.com/questions/212723/how-can-i-convert-lon-lat-coordinates-to-x-y

    :param ll: list of latitude/longitude values.
    :return: [x-coordinate, y-coordinate, z-coordinate] of each latitude/longitude value.
    """

    x = math.sin(math.pi / 2 - ll[0]) * math.cos(ll[1])
    y = math.sin(math.pi / 2 - ll[0]) * math.sin(ll[1])
    z = math.cos(math.pi / 2 - ll[0])
    return [x, y, z]


def process_coordinate(df, orig_df):
    """ Processes coordinates into multiple useful features

    :param df: pandas DataFrame containing coordinate data (unique values only)
    :param orig_df: pandas DataFrame containing the original coordinate column
    :return: pandas DataFrame consisting of latlong, zipcodes, country_codes, and xyz-coordinates
    """
    # Set up additional columns that will be returned alongside the transformed coordinates
    latlong = list(df.values.flatten())
    changes = {}
    changes_country = {}
    changes_xyz = {}
    changes_zipcode = {}

    # To check whether we are dealing with coordinate pairs or not
    single_coord = True

    # Instantiate geopy session to retrieve useful data
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    for i in range(len(latlong)):
        # Separate values into parts that are easier to work with
        val = re.split('([NSZEOW])', latlong[i])
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
    return orig_df


def process_day(df):
    """ Processes days into easy-to-encode strings. We assume that strings representing days are always at the front.

    :param df: pandas DataFrame containing strings representing days.
    :return: pandas DataFrame containing strings representing abbreviated days.
    """
    return df[df.columns[0]].str[:2]


def process_email(df):
    """ Processes e-mails into easy-to-encode strings with (possibly) enhanced properties.

    :param df: pandas DataFrame containing strings representing e-mails.
    :return: dict that maps the old values to the new values.
    """
    emails = list(df.values.flatten())
    changes = {}
    for i in range(len(emails)):
        val = emails[i].split("@", 1)
        val[1] = val[1].split(".", 1)[0]
        val = " ".join(val)
        changes[emails[i]] = val
    return changes


def process_filepath_url(df, type):
    """ Processes filepaths and URLs into easy-to-encode strings with (possibly) enhanced properties.

    :param df: pandas DataFrame containing strings representing filepaths or URLs.
    :param type: string containing whether the column contains 'filepath' or 'url' values.
    :return: pandas DataFrame containing easy-to-encode filepath strings.
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
    return changes


# TODO: change this probably when we adjust our pfsm
def process_month(df):
    """ Processes months into easy-to-encode strings. We assume that strings representing months are always at the
    front.

    :param df: pandas DataFrame containing strings representing months.
    :return: pandas DataFrame containing strings representing abbreviated months.
    """
    return df[df.columns[0]].str[:3]


def process_ordinal(df):
    """ Processes strings representing rank into ordinal-encoded numbers, adhering to the rank.

    :param df: pandas DataFrame containing strings representing ordered numbers.
    :return: dict that maps the old values to the new values.
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
    return changes


# TODO: add sentence/word embeddings to it as well
def process_sentence(df):
    """ Processes sentences into a list of nouns for easier encoding.

    :param df: pandas DataFrame containing sentences.
    :return: dict that maps the old values to the new values.
    """
    sentences = list(df.values.flatten())
    changes = {}
    for sent in sentences:
        # Using straight-up nltk word tokenizer
        tokenized = nltk.word_tokenize(sent)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos == 'NN']
        changes[sent] = " ".join(nouns)
    return changes


def process_zipcode(df, orig_df):
    """ Processes coordinates into multiple useful features.

    :param df: pandas DataFrame containing strings representing zipcodes.
    :param orig_df: pandas DataFrame containing the original zipcode column.
    :return: pandas DataFrame consisting of zipcodes, latlong, country_codes, and xyz-coordinates.
    """
    zipcodes = list(df.values.flatten())

    # Instantiate geopy session
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    # Create new dictionary for each feature
    changes = {}
    changes_latlong = {}
    changes_xyz = {}
    changes_country = {}
    changes_city = {}

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
            city = 'Unknown'

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
    return orig_df


def run(df, stringtype):
    """ Handles the processing of data in a unique string column into easy-to-encode (+ enhanced) data.

    :param df: pandas DataFrame containing the unique string column.
    :param stringtype: string indicating which unique string column df is.
    :return: pandas DataFrame consisting of column(s) containing processed (+ additional) data.
    """
    if stringtype in ['day', 'month']:
        return eval('process_{}(df)'.format(stringtype))
    else:
        df_unique = pd.DataFrame(df[df.columns[0]].unique(), columns=[df.columns[0]])
        if stringtype in ['filepath', 'url']:
            result = process_filepath_url(df_unique, stringtype)
        elif stringtype in ['coordinate', 'zipcode']:
            return eval('process_{}(df_unique, df)'.format(stringtype))
        else:
            result = eval('process_{}(df_unique)'.format(stringtype))
        return df.replace({df.columns[0]: result})
