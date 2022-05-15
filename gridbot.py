#!/usr/bin/env python3
"""Cyberjunky's 3Commas bot helpers."""
import argparse
import configparser
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from helpers.logging import Logger, NotificationHandler
from helpers.misc import wait_time_interval
from helpers.threecommas import init_threecommas_api

def set_api(api,logger):
    """Set API key and secret."""
    api=api
    logger=logger


    logger.error("Error setting API key: %s" % error["msg"])
    return error["msg"]

def load_config(datadir, program):
    """Create default or load existing config file."""

    cfg = configparser.ConfigParser()
    if cfg.read(f"{datadir}/{program}.ini"):
        return cfg

    cfg["settings"] = {
        "timezone": "Europe/Amsterdam",
        "timeinterval": 3600,
        "debug": False,
        "logrotate": 7,
        "3c-apikey": "Your 3Commas API Key",
        "3c-apisecret": "Your 3Commas API Secret",
        "notifications": False,
        "notify-urls": ["notify-url1", "notify-url2"],
    }

    cfg["gridbots_redbag_example"] = {
        "botids": [12345, 67890],
        "mode": "redbag",
    }

    cfg["gridbots_trade_example"] = {
        "botids": [12345, 67890],
        "mode": "trade",
    }

    with open(f"{datadir}/{program}.ini", "w") as cfgfile:
        cfg.write(cfgfile)

    return None


def strtofloat(txtstr):
    """Convert text string to float."""
    val = txtstr.text.strip()
    price = val.replace(".", "")
    floatval = price.replace(",", ".")

    return floatval


def get_gridbots_data(logger,pair):
    """Get the best gridbot settings from grid-bots.com."""

    url = "https://www.grid-bots.com"
    coin = pair.split("_")[1]

    griddata = {}
    try:
        result = requests.get(url)
        result.raise_for_status()
        soup = BeautifulSoup(result.text, features="html.parser")
        tablerows = [t for t in soup.find_all("tr") if not t.find_all("table")]

        for row in tablerows:
            rowcolums = row.find_all("td")
            if len(rowcolums) > 0:
                gridcoin = rowcolums[0].text.strip()
                if coin == gridcoin:
                    griddata["lower"] = strtofloat(rowcolums[2])
                    griddata["upper"] = strtofloat(rowcolums[3])
                    griddata["numgrid"] = int(rowcolums[4].text.strip())
                    griddata["tokensgrid"] = strtofloat(rowcolums[5])
                    break

        logger.debug(griddata)
    except requests.exceptions.HTTPError as err:
        logger.error("Fetching grid-bots data failed with error: %s" % err)
        return griddata

    logger.info("Fetched grid-bots data OK")

    return griddata


def update_gridbot(gridbot, upperprice, lowerprice):
    """Update gridbot with new grid."""

    botname = gridbot["name"]
    pair = gridbot["pair"]

    error, data = api.request(
        entity="grid_bots",
        action="manual_update",
        action_id=str(gridbot["id"]),
        payload={
            "bot_id": gridbot["id"],
            "name": gridbot["name"],
            "account_id": gridbot["account_id"],
            "pair": gridbot["pair"],
            "upper_price": upperprice,
            "lower_price": lowerprice,
            "quantity_per_grid": gridbot["quantity_per_grid"],
            "grids_quantity": gridbot["grids_quantity"],
            "leverage_type": gridbot["leverage_type"],
            "leverage_custom_value": gridbot["leverage_custom_value"]
        },
    )
    if data:
        logger.info(
            f"Moved the grid of gridbot '{botname}' using pair {pair} with"
            f" upper and lower price: {upperprice} - {lowerprice}",
            True,
        )
        return None

    logger.error(
        f"Error occurred updating gridbot '{botname}' with new upper price"
        f" and lower price of {upperprice} & {lowerprice} : %s" % error["msg"]
    )
    return error["msg"]


def update_gridbot_activelines(gridbot, maxactivebuylines, maxactiveselllines):
    """Update gridbot with new active line settings."""

    botname = gridbot["name"]
    pair = gridbot["pair"]

    error, data = api.request(
        entity="grid_bots",
        action="manual_update",
        action_id=str(gridbot["id"]),
        payload={
            "bot_id": gridbot["id"],
            "name": gridbot["name"],
            "account_id": gridbot["account_id"],
            "pair": gridbot["pair"],
            "upper_price": gridbot["upper_price"],
            "lower_price": gridbot["lower_price"],
            'max_active_buy_lines': maxactivebuylines,
            'max_active_sell_lines': maxactiveselllines,
            "quantity_per_grid": gridbot["quantity_per_grid"],
            "grids_quantity": gridbot["grids_quantity"],
        },
    )
    if data:
        logger.info(
            f"Set active lines of gridbot '{botname}' to"
            f" buy: {maxactivebuylines} and sell: {maxactiveselllines}",
            True,
        )
        return None

    logger.error(
        f"Error occurred updating gridbot '{botname}' with new active lines"
        f" buy: {maxactivebuylines} and sell: {maxactiveselllines}: %s" % error["msg"]
    )
    return error["msg"]

def bot_activate (botid):
    """Activate or deactivate a gridbot."""
    bot=str(botid["id"])
    error, data = api.request(
        entity="grid_bots",
        action='enable',
        action_id=str(botid["id"]),
    )
    if data:
        logger.info(f"Activated gridbot '{bot}'", True)
        return None

    logger.error(f"Error occurred activating gridbot : %s" % error["msg"])
    return error["msg"]

def manage_gridbot(thebot, upper_price, lower_price):
    """Move grid to match pricing."""
    botname = thebot["name"]

    # bot values to calculate with
    pair = thebot["pair"]
    upperprice = thebot["upper_price"]
    lowerprice = thebot["lower_price"]
    quantitypergrid = thebot["quantity_per_grid"]
    gridsquantity = thebot["grids_quantity"]
    strategytype = thebot["strategy_type"]
    currentprice = thebot["current_price"]

    logger.info("Current settings for '%s':" % botname)
    logger.info("Pair: %s" % pair)
    logger.info("Upper price: %s" % upperprice)
    logger.info("Lower price: %s" % lowerprice)
    logger.info("Quantity per grid: %s" % quantitypergrid)
    logger.info("Grid quantity: %s" % gridsquantity)
    logger.info("Strategy type: %s" % strategytype)
    logger.info("Current price for %s is %s" % (pair, currentprice))

    #gridinfo = get_gridbots_data(pair)

    #if gridinfo is None:
    #    logger.info(f"No grid setup information found for {pair}, skipping update")
    #   return

    newupperprice = float(upper_price) #gridinfo["upper"]
    newlowerprice = float(lower_price) #gridinfo["lower"]
    # newtokensgrid = gridinfo["tokensgrid"]
    # newnumgrid = gridinfo["numgrid"]

    # Test updating active lines for @IamtheOnewhoKnocks:
    # maxactivebuylines = 3
    # maxactiveselllines = 3

    # update_gridbot_activelines(thebot, maxactivebuylines, maxactiveselllines)

    if float(upperprice) == float(newupperprice):
        logger.info(
            f"Grid of gridbot '{botname}' with pair {pair}\nis already using"
            " correct upper and price, skipping update.",
            True,
        )
        return

    logger.info(
        f"Grid of gridbot '{botname}' with pair {pair} will be adjusted like this:\n"
        f"Upper: {upperprice} -> {newupperprice} Lower: {lowerprice} -> {newlowerprice}",
        True,
    )
    #return

    # Update the bot with new limits
    result = update_gridbot(thebot, newupperprice, newlowerprice)    
    if result and "Upper price should be at least " in result:
        uprice = re.search("Upper price should be at least ([0-9.]+)", result)
        if uprice:
            upperprice = uprice[1]
            logger.info(
                f"New upper price was not accepted, retrying with suggested minimum price of {upperprice}"
            )
            result = update_gridbot(thebot, upperprice, newlowerprice)
            if result:
                logger.error(
                    f"Failed to update gridbot with suggested minimum upper price of {upperprice}"
                )
    bot_activate (thebot)


datadir = os.getcwd()
api =init_threecommas_api(load_config(datadir, '_gridbot'))

program ='_gridbot'
# Initialize 3Commas API

# Create or load configuration file
config = load_config(datadir, '_gridbot')

if not config:
    # Initialise temp logging
    logger = Logger(datadir, '{program}.ini', None, 7, False, False)
    logger.info(
        f"Created example config file '{datadir}/{program}, edit it and restart the program"
    )
    sys.exit(0)
else:
    # Handle timezone
    if hasattr(time, "tzset"):
        os.environ["TZ"] = config.get(
            "settings", "timezone", fallback="Europe/Amsterdam"
        )
        time.tzset()

    # Init notification handler
    notification = NotificationHandler(
        '_gridbot.ini',
        config.getboolean("settings", "notifications"),
        config.get("settings", "notify-urls"),
    )

    # Initialise logging
    logger = Logger(
        datadir,
        '_gridbot.ini',
        notification,
        int(config.get("settings", "logrotate", fallback=7)),
        config.getboolean("settings", "debug"),
        config.getboolean("settings", "notifications"),
    )