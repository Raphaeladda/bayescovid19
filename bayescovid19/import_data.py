#!/usr/bin/env python3
___authors___ = 'N. Fouqueray, E. Vazquez'
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

"""Import data
Sources

* data.gouv.fr / Santé Publique France

  https://www.data.gouv.fr/fr/datasets/donnees-relatives-a-lepidemie-du-covid-19/#_

* ECDC

"""
import urllib.request
from datetime import datetime
from os import getcwd
from os.path import join

import numpy as np
import pandas as pd


class Data():
    "Stores url and filenames"

    def __init__(self, url, filename):
        self.url = url
        self.filename = filename


def download_data(d, debug=False):
    # Download data
    import ssl

    # This circumvents the SSL certificate problem
    ssl._create_default_https_context = ssl._create_unverified_context
    if debug: print('Saving %s at %s' % (d.filename, d.url))
    urllib.request.urlretrieve(d.url, d.filename)


def get_data(filename, separator):
    # Extract data
    dataframe = pd.read_csv(filename, sep=separator, error_bad_lines=False)
    return dataframe


def get_fatalities(country='France', reuse=False, debug=False):
    # Initialization

    if debug: print('Import data...')
    root = getcwd()
    datapath = join(root, 'data')

    # Source defintions

    download_JHU = False

    datasources = []

    url_data_01 = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    filename_01 = join(datapath, 'data_ECDC_deaths.csv')
    datasources.append(Data(url_data_01, filename_01))

    if country == 'France':
        url_data_02 = 'http://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7'
        # ancienne adresse 'https://www.data.gouv.fr/fr/datasets/r/b94ba7af-c0d6-4055-a883-61160e412115'
        filename_02 = join(datapath, 'data_SPF_A.csv')
        datasources.append(Data(url_data_02, filename_02))


    if download_JHU:
        url_data_03 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        filename_03 = join(datapath, 'data_JHU_deaths.csv')
        datasources.append(Data(url_data_03, filename_03))

    if debug: print('Download data...')
    if not reuse:
        for data in datasources:
            download_data(data, debug)

    
    # --- Data ECDC.
    file = datasources[0].filename
    if debug: print('Reading %s...' % file)
    df_ECDC = get_data(file, ',')
    
    if debug: print('Number of lines: %d' % df_ECDC.shape[0])
    df_ECDC_country = df_ECDC.loc[df_ECDC.loc[:]['countriesAndTerritories'] == country]
    df_ECDC_country = df_ECDC_country.copy()
    df_ECDC_country = df_ECDC_country.drop(
        columns=['day', 'month', 'year', 'countryterritoryCode', 'geoId', 'countryterritoryCode', 'popData2018'])
    n = df_ECDC_country.shape[0]
    if debug: print(df_ECDC_country.head(10))

    
    # --- Data Santé Publique France
    if country == 'France':
        file = datasources[1].filename
        if debug: print('Reading %s...' % file)
        df_SPF = get_data(file, ';')
        if debug:
            print('Number of lines: %d' % df_SPF.shape[0])
            print('\n---')
            print(df_SPF.head(10))
            print('...\n---\n')

        departements = df_SPF.dep.unique()
        dates = df_SPF.jour.unique()

        
    # --- Data JH Univ.
    if download_JHU:
        # Pour le dataframe portant sur le nombre de décès
        file = datasources[2].filename
        if debug: print('Reading %s...' % file)
        df_JHU = get_data(file, ',')
        if debug: print('Number of lines: %d' % df_JHU.shape[0])

        # Extraction des données France
        df_JHU_country = df_JHU.loc[df_JHU.loc[:]['Country/Region'] == 'France']
        df_JHU_country_Metrop = df_JHU_country.loc[pd.isna(df_JHU_country.loc[:]['Province/State'])]
        df_JHU_country_Metrop = df_JHU_country_Metrop.copy()

        df_JHU_country_Metrop.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'], inplace=True)
        df_JHU = df_JHU_country_Metrop.transpose()

        df_JHU.columns = ['Fatalities']
        if debug: print(df_JHU.head(10))

        n = df_JHU.shape[0]


    # --- Fusion données
    # construction d'un df "décès" avec les dates en indices et les
    # départements en colonnes + agrégation de tous les départements

    # extraction des dates
    dates_ECDC = df_ECDC_country['dateRep']
    dates = [datetime.strptime(date, '%d/%m/%Y') for date in dates_ECDC]

    df_dates = pd.DataFrame(dates)
    df_dates.columns = ['date']

    # extract deaths in ECDC data
    if country == 'France':
        n_addtional_columns = 96
    else:
        n_addtional_columns = 0
    
    fatalities = np.empty([n, 1 + n_addtional_columns])
    fatalities[:] = np.NaN
    fatalities[:, 0] = df_ECDC_country.deaths
    df_fatalities = pd.DataFrame(fatalities)

    # SPF
    if country == 'France':
        dpt_list = pd.unique(df_SPF['dep']).tolist()
        dpt_list = [dpt_list[i] for i in range(n_addtional_columns)]
    else:
        dpt_list = []
        
    dpt_list = ['total'] + dpt_list
    df_fatalities.columns = dpt_list

    df_fatalities = pd.concat([df_dates, df_fatalities], axis=1)

    if country == 'France':
        # insertion des décès par département
        for i in range(df_SPF.shape[0]):
            sexe = df_SPF.sexe[i]
            if sexe == 0:
                date = datetime.strptime(df_SPF.jour[i], '%Y-%m-%d')
                dep = df_SPF.dep[i]
                dc = df_SPF.dc[i]
                td = df_fatalities[df_fatalities.date == date]
                if not td.empty and dep in dpt_list:
                    j = td.index[0]
                    df_fatalities.loc[j, dep] = int(dc)

    # remove all stats before 1/1/2020
    idx = (df_fatalities.date >= pd.Timestamp(2020, 1, 1))
    df_fatalities = df_fatalities[idx]

    # ECDC data are stored by decreasing dates: reverse
    df_fatalities = df_fatalities.iloc[::-1]
    df_fatalities.index = range(df_fatalities.shape[0])

    # cumulate the daily fatalities
    df_fatalities['total'] = df_fatalities['total'].cumsum()

    return df_fatalities
