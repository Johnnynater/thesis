import math
import nltk
import re
import os
import numpy as np
import pandas as pd
from functools import partial
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
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


def remove_longest_fix(values, suf=False):
    """ Remove the longest common suffix or prefix of a List of Strings by abusing os.path.commonprefix().

    :param values: a List of Strings.
    :param suf: a Boolean representing whether the suffix needs to be removed, default=False.
    :return: a List of Strings without the longest common prefix / suffix.
    """
    if suf:
        values = [x[::-1] for x in values]

    com_fix = os.path.commonprefix(values)
    new_values = [x[len(com_fix):] for x in values]

    if suf:
        new_values = [x[::-1] for x in new_values]
    return new_values


def process_boolean(df):
    """ Process boolean-esque strings into numerical representations.

    :param df: a pandas DataFrame consisting of Strings that represent boolean data.
    :return: a pandas DataFrame consisting of encoded Boolean values.
    """
    le = LabelEncoder()
    for col in df:
        df[col] = le.fit_transform(df[col])
    return df, 2


def process_coordinate(df, orig_df, dense):
    """ Process coordinates into multiple useful features.

    :param df: a pandas DataFrame consisting of Strings that represent coordinate data (unique values only).
    :param orig_df: a pandas DataFrame consisting of the original coordinate column.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a pandas DataFrame consisting of latlong, zipcodes, country_codes, and xyz-coordinates;
             a (List of) Integer(s) indicating the encoding type(s) required for each column.
    """
    # Set up additional columns that will be returned alongside the transformed coordinates
    latlong = list(df.values.flatten())

    changes, changes_country, changes_xyz, changes_zipcode = {}, {}, {}, {}
    changes_lat, changes_long = {}, {}
    changes_x, changes_y, changes_z = {}, {}, {}

    # To check whether we are dealing with coordinate pairs or not
    single_coord = True

    # Instantiate geopy session to retrieve useful data
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    for i in range(len(latlong)):
        # Separate values into parts that are easier to work with
        val = re.split('([NSZEOW])', latlong[i])
        val = [re.split(r'[*°º\'"´dms\\ ]', val[j]) for j in range(len(val))]
        val = list(filter(lambda a: a != [''], val))
        val = [list(filter(lambda a: a != '', item)) for item in val]

        if not val[0][0].isdigit():
            for j in range(len(val)):
                val[j] = val[j][0].split('.')

        # We have to process coordinates with just north/south or east/west differently, also put any letters upfront
        if len(val) == 2:
            # Swap entries in case our format is [[digits],[letter]]
            if val[0][0].isdigit():
                val[0], val[1] = val[1], val[0]
            val = [val[0] + val[1]]
        elif len(val) == 4:
            # Swap entries in case our format is [[digits],[letter]]
            if val[0][0].isdigit():
                val[0], val[1], val[2], val[3] = val[1], val[0], val[3], val[2]
            val = [val[0] + val[1], val[2] + val[3]]

        # Convert coordinates into latitude/longitude values, inspired by
        # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
        for j in range(len(val)):
            val[j][3] = val[j][3].replace(',', '.')
            if val[j][0] in 'SZW':
                val[j] = -(float(val[j][1]) + float(val[j][2]) / 60 + float(val[j][3]) / 3600)
            else:
                val[j] = float(val[j][1]) + float(val[j][2]) / 60 + float(val[j][3]) / 3600

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
            changes_zipcode[latlong[i]] = zipcode
            if dense:
                changes_xyz[latlong[i]] = xyz
            else:
                changes_x[latlong[i]] = xyz[0]
                changes_y[latlong[i]] = xyz[1]
                changes_z[latlong[i]] = xyz[2]

        # Insert obtained latlong into the corresponding dict
        if not dense and len(val) == 2:
            changes_lat[latlong[i]] = val[0]
            changes_long[latlong[i]] = val[1]
        else:
            changes[latlong[i]] = val[0]

    # Replace the original values for the encoded ones
    if not dense and not single_coord:
        orig_df[str(df.columns[0]) + '_lat'] = orig_df[df.columns[0]].map(changes_lat)
        orig_df[str(df.columns[0]) + '_long'] = orig_df[df.columns[0]].map(changes_long)
        orig_df = orig_df.drop(columns=[df.columns[0]])
    else:
        orig_df[df.columns[0]] = orig_df[df.columns[0]].map(changes)

    if not single_coord:
        # In case we are dealing with coordinate pairs, add the additional info
        orig_df[str(df.columns[0]) + '_country'] = orig_df[df.columns[0]].map(changes_country)
        orig_df[str(df.columns[0]) + '_zipcode'] = orig_df[df.columns[0]].map(changes_zipcode)
        if dense:
            orig_df[str(df.columns[0]) + '_xyz'] = orig_df[df.columns[0]].map(changes_xyz)
        else:
            orig_df[str(df.columns[0]) + '_x'] = orig_df[df.columns[0]].map(changes_x)
            orig_df[str(df.columns[0]) + '_y'] = orig_df[df.columns[0]].map(changes_y)
            orig_df[str(df.columns[0]) + '_z'] = orig_df[df.columns[0]].map(changes_z)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    if single_coord:
        return orig_df, 2
    elif dense:
        return orig_df, [2, 1, 1, 2]
    else:
        return orig_df, [2, 2, 1, 1, 2, 2, 2]


def process_date(df):
    """ Process datetime types from ptype into unix epoch (was the easiest for the remaining time).

    :param df: a pandas DataFrame consisting of Datetime-like Strings.
    :return: a pandas DataFrame consisting of Integers representing UNIX-timestamps of each DateTime value;
             an Integer representing which encoding needs to be applied.
    """
    for col in df:
        df[col] = df[col].apply(lambda x: (pd.to_datetime(x, errors='ignore', infer_datetime_format=True)))
        df[col] = (pd.to_datetime(df[col]).values.astype(float) / 10 ** 9).astype(int)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return df, 2


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

    # Remove common suffix
    emails = remove_longest_fix(emails, True)

    changes = {}

    for i in range(len(emails)):
        # Split between name and domain of e-mail + remove top-level domains
        val = emails[i].split("@", 1)
        if len(val) > 1:
            val[1] = val[1].split(".", 1)[0]
        val = " ".join(val)
        changes[emails[i]] = val

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 1


def process_filepath_url(df, featuretype):
    """ Process filepaths and URLs into easy-to-encode strings with (possibly) enhanced properties.

    :param df: a pandas DataFrame consisting of Strings that represent filepaths or URLs.
    :param featuretype: a String representing whether the column contains 'filepath' or 'url' entries.
    :return: a pandas DataFrame consisting of easy-to-encode filepath or URL Strings;
             an Integer representing the encoding type required.
    """
    data = list(df.values.flatten())

    # Remove common prefix / suffix
    fixless_data = remove_longest_fix(remove_longest_fix(data), True)

    changes = {}
    for i in range(len(data)):
        val = fixless_data[i]
        if featuretype == 'url':
            if '://' in fixless_data[i]:
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
        'jan': ['01', 1, 31],  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        'feb': ['02', 2, 28],  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        'mar': ['03', 3, 31],  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        'apr': ['04', 4, 30],  # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        'may': ['05', 5, 31],  # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        'jun': ['06', 6, 30],  # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        'jul': ['07', 7, 31],  # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        'aug': ['08', 8, 31],  # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        'sep': ['09', 9, 30],  # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        'oct': ['10', 10, 31],  # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        'nov': ['11', 11, 30],  # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        'dec': ['12', 12, 31]  # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    for i in range(len(months)):
        # Set the dd/mmm/yy variables and split the data based on characters in each value
        dd, mmm, yy = '', '', ''
        split_vals = re.split('[,.\-_ ]', months[i])
        split_vals = list(filter(lambda a: a != "", split_vals))
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
                if not val.isdigit():
                    if val[0] == '\'':
                        yy = val
                    else:
                        mmm = val
            if yy:
                split_vals.remove(yy)
                yy = yy[1:]
            if mmm:
                split_vals.remove(mmm)
                mmm = mmm[:3]
            for val in split_vals:
                if not val.isdigit():
                    if val[0] == '\'':
                        yy = val[1:]
                    else:
                        mmm = val[:3]
                else:
                    if len(val) == 4:
                        yy = val[2:]
                    elif int(val) > month_to_int[mmm.lower()][2] or dd:
                        yy = val
                    else:
                        dd = val if len(val) == 2 else '0' + str(val)
            changes[months[i]] = int(yy + month_to_int[mmm.lower()][0] + dd)

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 2


def process_numerical(df, orig_df, dense):
    """ Convert numerical strings (such as ranges, units, etc.) into ordinally-encoded data.

    :param df: a pandas DataFrame consisting of Strings that represent numerical values.
    :param orig_df: a pandas DataFrame consisting of Strings that represent the original numerical column.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a Dictionary that maps the old values to the new values (String -> Integer);
             an Integer representing the encoding type required.
    """
    nums = list(df.values.flatten())
    processed_nums = []
    num_range = False

    for i in range(len(nums)):
        if re.fullmatch('[0-9]+ ?[-:_/(to)] ?[0-9]+', nums[i]):
            # Take the mean of the range
            num_range = True
            digits = int(np.mean([int(s) for s in re.split(' ?[-:_/(to)] ?', nums[i]) if s.isdigit()]))
        elif re.fullmatch(
                r"(([<>$#@%=]+|(([Ll])ess|([Ll])ower|([Gg])reater|([Hh])igher) than"
                r"|(([Uu])nder|([Bb])elow|([Oo])ver|([Aa])bove))[ \-]?[0-9]+)|([0-9]+ ?[<>+$%=]+)",
                nums[i]
        ):
            # Only take the relevant numbers, but set num_range to true
            num_range = True
            digits = int(''.join([s for s in nums[i] if s.isdigit()]))
        else:
            # Only take the relevant numbers
            digits = re.split(r" ?[+:;&'] ?", nums[i])

        processed_nums.append([nums[i], digits])

    if num_range:
        # Sort the data based on the processed numbers
        processed_nums.sort(key=lambda x: x[1])

        # Map the original values to the (ordinal encoded) values
        orig_df[str(df.columns[0])] = orig_df[str(df.columns[0])].map(
            {processed_nums[i][0]: i for i in range(len(processed_nums))}
        )
    else:
        if not dense:
            # Assumption is made that all list containing processed numbers are of equal length
            for i in range(len(processed_nums[0][1])):
                orig_df[str(df.columns[0]) + str(i)] = orig_df[str(df.columns[0])].map(
                    {item[0]: int(item[1][i]) for item in processed_nums}
                )
            orig_df = orig_df.drop(columns=[df.columns[0]])
            return orig_df, [2 for _ in range(len(processed_nums[0][1]))]
        else:
            orig_df[str(df.columns[0])] = orig_df[str(df.columns[0])].map(
                {item[0]: item[1] for item in processed_nums}
            )

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return orig_df, 2


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
        if nouns:
            changes[sent] = " ".join(nouns)
        else:
            changes[sent] = sent

    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    return changes, 1


def process_zipcode(df, orig_df, dense):
    """ Process coordinates into multiple useful features.

    :param df: a pandas DataFrame consisting of Strings that represent zipcodes.
    :param orig_df: a pandas DataFrame consisting of Strings that represent the original zipcode column.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a pandas DataFrame consisting of zipcodes, latlong, country_codes, and xyz-coordinates;
             a (List of) Integers(s) indicating the encoding type(s) required for each column.
    """
    # Instantiate geopy session
    geolocator = Nominatim(user_agent='my-app')
    reverse = partial(geolocator.reverse)

    # Flatten dataframe to list + create new dictionary for each feature
    zipcodes = list(df.values.flatten())
    changes, changes_city, changes_country, changes_latlong, changes_xyz = {}, {}, {}, {}, {}
    changes_lat, changes_long = {}, {}
    changes_x, changes_y, changes_z = {}, {}, {}

    for i in range(len(zipcodes)):
        # Remove any spaces (this is basically all the pre-processing we need for just zipcodes)
        zip_info = geolocator.geocode(zipcodes[i])
        if zip_info:
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
        else:
            city = 'unknown'
            country = 'unknown'
            latlong = [0, 0]
            xyz = [0, 0, 0]

        # Insert the obtained values in the corresponding dicts
        changes_city[zipcodes[i]] = city
        changes_country[zipcodes[i]] = country
        if dense:
            changes_latlong[zipcodes[i]] = latlong
            changes_xyz[zipcodes[i]] = xyz
        else:
            changes_lat[zipcodes[i]] = latlong[0]
            changes_long[zipcodes[i]] = latlong[1]
            changes_x[zipcodes[i]] = xyz[0]
            changes_y[zipcodes[i]] = xyz[1]
            changes_z[zipcodes[i]] = xyz[2]

    # Map the updated + new values to the corresponding values in the original DataFrame
    # 0 = ordinal encoding, 1 = nominal encoding, 2 = no encoding needed / already encoded
    orig_df[str(df.columns[0]) + '_city'] = orig_df[df.columns[0]].map(changes_city)
    orig_df[str(df.columns[0]) + '_country'] = orig_df[df.columns[0]].map(changes_country)
    if dense:
        orig_df[str(df.columns[0]) + '_latlong'] = orig_df[df.columns[0]].map(changes_latlong)
        orig_df[str(df.columns[0]) + '_xyz'] = orig_df[df.columns[0]].map(changes_xyz)
        return orig_df, [1, 1, 1, 2, 2]
    else:
        orig_df[str(df.columns[0]) + '_lat'] = orig_df[df.columns[0]].map(changes_lat)
        orig_df[str(df.columns[0]) + '_long'] = orig_df[df.columns[0]].map(changes_long)
        orig_df[str(df.columns[0]) + '_x'] = orig_df[df.columns[0]].map(changes_x)
        orig_df[str(df.columns[0]) + '_y'] = orig_df[df.columns[0]].map(changes_y)
        orig_df[str(df.columns[0]) + '_z'] = orig_df[df.columns[0]].map(changes_z)
        return orig_df, [1, 1, 1, 2, 2, 2, 2, 2]


def run(df, stringtype, dense):
    """ Handle the processing of data in a unique string column into easy-to-encode (+ enhanced) data.

    :param df: a pandas DataFrame consisting of unique String entries.
    :param stringtype: a String indicating which unique String type the column df consists of.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a pandas DataFrame consisting of column(s) containing processed (+ additional) data; a (List of) Integer(s)
             indicating the encoding type(s) required for each column.
    """
    if stringtype in ['day']:
        return eval('process_{}(df)'.format(stringtype))
    elif stringtype in ['boolean', 'gender']:
        return process_boolean(df)
    elif stringtype in ['date-iso-8601', 'date-eu']:
        return process_date(df)
    else:
        df_unique = pd.DataFrame(df[df.columns[0]].unique(), columns=[df.columns[0]])
        if stringtype in ['filepath', 'url']:
            result, encode = process_filepath_url(df_unique, stringtype)
        elif stringtype in ['coordinate', 'numerical', 'zipcode']:
            result, encode = eval('process_{}(df_unique, df, dense)'.format(stringtype))
            return result, encode
        else:
            result, encode = eval('process_{}(df_unique)'.format(stringtype))
        return df.replace(result), encode
