import re
import math
import nltk
import pandas as pd
from functools import partial
from geopy.geocoders import Nominatim
from word2number import w2n


def process_coordinates(df):
    """ Processes coordinates into multiple useful features

    :param df: pandas DataFrame containing coordinate data
    :return: pandas DataFrame consisting of latlong, zipcodes, country_codes, and xyz-coordinates
    """
    # Set up additional columns that will be returned alongside the transformed coordinates
    latlong = list(df.values.flatten())
    zipcode = []
    country = []
    latlong_xyz = []

    # Instantiate geopy session to retrieve useful data
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    for i in range(len(latlong)):
        # Separate values into parts that are easier to work with
        latlong[i] = re.split('([NSZEOW])', latlong[i])
        latlong[i] = latlong[i][1:]
        latlong[i] = [latlong[i][j].split('.') for j in range(len(latlong[i]))]

        # We have to process coordinates with just north/south or east/west differently
        if len(latlong[i]) == 2:
            latlong[i] = [latlong[i][0] + latlong[i][1]]
        elif len(latlong[i]) == 4:
            latlong[i] = [latlong[i][0] + latlong[i][1], latlong[i][2] + latlong[i][3]]

        # Convert coordinates into latitude/longitude values, inspired by
        # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
        for j in range(len(latlong[i])):
            if latlong[i][j][0] in 'SZW':
                latlong[i][j] = -(int(latlong[i][j][1]) + int(latlong[i][j][2]) / 60 + int(latlong[i][j][3]) / 3600)
            else:
                latlong[i][j] = int(latlong[i][j][1]) + int(latlong[i][j][2]) / 60 + int(latlong[i][j][3]) / 3600

        if len(latlong[i]) == 2:
            # Convert (GPS) coordinates to (X, Y, Z) coordinates. Inspired by Nat Wilson's post at
            # https://gis.stackexchange.com/questions/212723/how-can-i-convert-lon-lat-coordinates-to-x-y
            x = math.sin(math.pi / 2 - latlong[i][0]) * math.cos(latlong[i][1])
            y = math.sin(math.pi / 2 - latlong[i][0]) * math.sin(latlong[i][1])
            z = math.cos(math.pi / 2 - latlong[i][0])
            latlong_xyz.append([x, y, z])

            # Convert coordinates to additional data using geopy
            latlong_info = reverse('{}, {}'.format(latlong[i][0], latlong[i][1]))

            # It can be the case that the latlong values point to a deserted place. In this case, there is no zipcode
            # or country_code and we mark those places as 'unknown'
            if latlong_info:
                if 'postcode' in latlong_info.raw['address']:
                    zipcode.append(latlong_info.raw['address']['postcode'])
                else:
                    zipcode.append('unknown')
                if 'country_code' in latlong_info.raw['address']:
                    country.append(latlong_info.raw['address']['country_code'])
                else:
                    country.append('unknown')
            else:
                zipcode.append('unknown')
                country.append('unknown')

            return pd.DataFrame(
                {
                    df.columns[0] + '_latlong': latlong,
                    df.columns[0] + '_zipcode': zipcode,
                    df.columns[0] + '_country': country,
                    df.columns[0] + '_xyz': latlong_xyz
                }
            )
        else:
            # In case we are dealing with only north/south or east/west values, we cannot use geopy
            return pd.DataFrame(
                {
                    df.columns[0] + '_latlong': latlong
                }
            )


def process_days(df):
    """ Processes days into easy-to-encode strings. We assume that strings representing days are always at the front.

    :param df: pandas DataFrame containing strings representing days.
    :return: pandas DataFrame containing strings representing abbreviated days.
    """
    return df[df.columns[0]].str[:2]


def process_emails(df):
    """ Processes e-mails into easy-to-encode strings with (possibly) enhanced properties.

    :param df: pandas DataFrame containing strings representing e-mails.
    :return: pandas DataFrame containing easy-to-encode e-mail strings.
    """
    emails = list(df.values.flatten())
    for i in range(len(emails)):
        at_sep = emails[i].split("@", 1)
        at_sep[1] = at_sep[1].split(".", 1)[0]
        emails[i] = " ".join(at_sep)
    return pd.DataFrame({df.columns[0]: emails})


def process_filepaths_urls(df, url):
    """ Processes filepaths and URLs into easy-to-encode strings with (possibly) enhanced properties.

    :param df: pandas DataFrame containing strings representing filepaths or URLs.
    :return: pandas DataFrame containing easy-to-encode filepath strings.
    """
    data = list(df.values.flatten())
    for i in range(len(data)):
        if url:
            if '://' in data[i]:
                data[i] = data[i].split("://", 1)[1]
        data[i] = re.split("[^a-zA-Z0-9]", data[i])
        data[i] = list(filter(lambda a: a != "", data[i]))
        data[i] = " ".join(data[i])
    return pd.DataFrame({df.columns[0]: data})


# TODO: change this probably when we adjust our pfsm
def process_months(df):
    """ Processes months into easy-to-encode strings. We assume that strings representing months are always at the
    front.

    :param df: pandas DataFrame containing strings representing months.
    :return: pandas DataFrame containing strings representing abbreviated months.
    """
    return df[df.columns[0]].str[:3]


def process_ordinals(df):
    """ Processes strings representing rank into ordinal-encoded numbers, adhering to the rank.

    :param df: pandas DataFrame containing strings representing ordered numbers.
    :return: pandas DataFrame containing integers.
    """
    ordinals = list(df.values.flatten())
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
    return pd.DataFrame({df.columns[0]: ordinals})


# TODO: add sentence/word embeddings to it as well
def process_sentences(df):
    """ Processes sentences into a list of nouns for easier encoding.

    :param df: pandas DataFrame containing sentences.
    :return: pandas DataFrame containing only the nouns that were present in each row.
    """
    sentences = list(df.values.flatten())
    keyword_list = []
    for sent in sentences:
        # Using straight-up nltk word tokenizer
        tokenized = nltk.word_tokenize(sent)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos == 'NN']
        keyword_list.append(" ".join(nouns))
    return pd.DataFrame({ df.columns[0] + '_nouns': keyword_list })


def process_zipcodes(df):
    """ Processes coordinates into multiple useful features

    :param df: pandas DataFrame containing strings representing zipcodes
    :return: pandas DataFrame consisting of zipcodes, latlong, country_codes, and xyz-coordinates
    """
    zipcodes = list(df.values.flatten())

    # Instantiate geopy session and additional columns
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    latlong = []
    latlong_xyz = []
    country = []
    city = []

    for i in range(len(zipcodes)):
        # Remove any spaces (this is basically all the pre-processing we need for just zipcodes)
        zipcodes[i] = zipcodes[i].replace(' ', '')
        zip_info = geolocator.geocode(zipcodes[i])
        latlong.append([zip_info.latitude, zip_info.longitude])

        # TODO: make separate function for this, bc code duplication with process_coordinates.
        # Convert latitude/longitude to xyz
        x = math.sin(math.pi / 2 - zip_info.latitude) * math.cos(zip_info.longitude)
        y = math.sin(math.pi / 2 - zip_info.latitude) * math.sin(zip_info.longitude)
        z = math.cos(math.pi / 2 - zip_info.latitude)
        latlong_xyz.append((x, y, z))

        # Gather more information using geopy
        latlong_info = reverse('{}, {}'.format(zip_info.latitude, zip_info.longitude)).raw
        country.append(latlong_info['address']['country_code'])
        if 'city' in latlong_info['address']:
            city.append(latlong_info['address']['city'])
        else:
            city.append('Unknown')
    return pd.DataFrame(
        {
            df.columns[0]: zipcodes,
            df.columns[0] + '_latlong': latlong,
            df.columns[0] + '_country': country,
            df.columns[0] + '_xyz': latlong_xyz
        }
    )


def run(df, stringtype):
    for name in df:
        print('')
    return
