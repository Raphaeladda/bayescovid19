#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, getopt, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

import bayescovid19.import_data as import_data


def plot_fatalities(df, cols, country='', cumulated=True, semilog=True, savefig=False):
    # Handle date time conversions between pandas and matplotlib
    register_matplotlib_converters()
    sns.set()

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(12, 6))

    x = df['date']

    for i in range(len(cols)):

        if cumulated:
            y = df[cols[i]]
            idx = y.between(1, 10e4)
            y_ = y[idx]
            y_last = y_.iloc[-1]
        else:
            y_ = df[cols[i]]
            y = np.concatenate(([0], np.diff(y_)))
            y_last = y_.iloc[-1]

        plot_label = cols[i] + ", n = %d" % y_last
        # Plot
        if cumulated:
            plt.plot(x, y, marker='o', label=plot_label)
            plt.text(x.iloc[-1] + pd.Timedelta('1 days'), y_last, cols[i])
        else:
            plt.stem(x, y, label=plot_label)

        if semilog:
            plt.yscale('log')

    ax.set(xlabel='Date', ylabel='Fatalities')
    if cumulated:
        ax.set(title='Daily cumulated fatalities for '+country)
    else:
        ax.set(title='Daily fatalities for '+country)

    # Define the date format
    date_form = mdates.DateFormatter("%b %d, %y")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1) 
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    left, right = plt.xlim()
    plt.xlim(mdates.date2num(datetime.date(2020, 2, 9)), right)
    plt.legend()
    # Show
    plt.show()
    if savefig:
        plt.savefig('fatalities.png')

def show(country='France'):
    # --- Données

    df_fatalities = import_data.get_fatalities(country)

    if country == 'France':

        # Création d'une liste des départements métropolitains (au format str)
        dpts = list(df_fatalities)
        dpts.remove('date')
        dpts.remove('total')

        # --- find largest
        dc = list(df_fatalities.iloc[-1])
        del (dc[0:2])
        del (dc[-1])
        idx = sorted(range(len(dc)), key=lambda k: dc[k])
        dpts_sorted = [dpts[i] for i in reversed(idx)]

        # Visualisation de l'évolution du nombre de décès par jour en France
        dpts_view = ['total'] + dpts_sorted[0:3] # + [dpts_sorted[20], '91', '77', '92']
        dpts_view = ['total', '33', '75', '68']

    else:
        dpts_view = ['total']

    # Cumulated deaths
    plot_fatalities(df_fatalities, dpts_view, country=country, cumulated=True)

    # Daily fatalities
    plot_fatalities(df_fatalities, ['total'], country=country, cumulated=False, semilog=False)


def main(argv):

    country = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["country="])
    except getopt.GetoptError:
        print('Usage: view_data.py -i <country_name>')
        sys.exit(2)

    if len(opts) == 0:
        print('Usage: view_data.py -i <country_name>')
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: view_data.py -i <country_name>')
            sys.exit()
        elif opt in ("-i", "--country"):
            country = arg

    print('Viewing data for '+country)

    show(country) 
   
if __name__ == "__main__":

   main(sys.argv[1:])
