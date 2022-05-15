import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
from prophet import Prophet
from datetime import date
import san
san.ApiConfig.api_key = 'l5xaofjydbsmnivh_t3ioip3icldqopkd'
from san.extras.event_study import event_study, signals_format, hypothesis_test
import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    symbol='flow'
    TF='1h'
    stime="2021-01-27"


    if TF !='1d':
        format='%Y-%m-%d %H:%M:%S'   
    else:
        format='%Y-%m-%d'

    from datetime import datetime, timedelta
    import ciso8601
    data=pd.DataFrame()
    get_df=pd.DataFrame()


    start_date=ciso8601.parse_datetime(stime)
    end_date=datetime.now()#- timedelta(days=120)
    step= 100
    numdays = (end_date- start_date).days 
    start_dt=start_date
    #for h in range(0,numdays,step
    h=0
    end_dt=start_dt
    while end_dt<datetime.now():   
        start_dt= datetime.now()- timedelta(days=numdays-h)   
        if start_dt>datetime.now():
            break
        end_dt=start_dt+timedelta(days=step) 
        if end_dt>datetime.now():
            end_dt=datetime.now()-timedelta(days=1)
        print ('Get FLOW price data from: ', start_dt, ' to: ', end_dt)
        print('Reqest #:',h,' Date: ',end_dt,' get days:',(end_dt-start_dt).days)
        get_df= san.get("ohlcv/"+symbol, 
        from_date=start_dt,
        to_date=end_dt,
        interval=TF).closePriceUsd
        data=pd.concat([data,get_df] )
        h=h+step
    data.columns=['Price']
    df=data
    df.index=pd.to_datetime(df.index).strftime(format)
    df['ds'] = pd.to_datetime(df.index)
    df_prohet=pd.DataFrame()
    df_no_index = df.reset_index(drop=True)
    df_prohet['y']=df_no_index['Price']
    df_prohet['ds']=df_no_index['ds']

    m = Prophet(interval_width=0.95, 
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=0.01
        )
    model = m.fit(df_prohet)

    future = m.make_future_dataframe(periods=24*3,freq=TF)
    forecast = m.predict(future)
    forecast2 = forecast.set_index ('ds')
    df_plt=df.set_index ('ds')
    y=df_plt['2021-01-01': ]['Price']
    df_plt=df.set_index ('ds')
    y=df_plt['2021-01-01': ]['Price']
    yhat=forecast2['2021-01-01': ]['yhat']
    yhat_upper=forecast2['2021-01-01': ]['yhat_upper']
    yhat_lowwer=forecast2['2021-01-01': ]['yhat_lower']
    # import plotly packages 
    import plotly.graph_objects as go
    import plotly.express as px


    # Python
    def stan_init(m):
        """Retrieve parameters from a trained model.
        
        Retrieve parameters from a trained model in the format
        used to initialize a new Stan model.
        
        Parameters
        ----------
        m: A trained model of the Prophet class.
        
        Returns
        -------
        A Dictionary containing retrieved parameters of m.
        
        """
        res = {}
        for pname in ['k', 'm', 'sigma_obs']:
            res[pname] = m.params[pname][0][0]
        for pname in ['delta', 'beta']:
            res[pname] = m.params[pname][0]
        return res

    import gridbot as Gb
    datadir = os.getcwd()
    program ='_gridbot'
    # Initialize 3Commas API

    # Create or load configuration file
    config = Gb.load_config(datadir, '_gridbot')
    if not config:
        # Initialise temp logging
        logger = Gb.Logger(datadir, '{program}.ini', None, 7, False, False)
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
        notification = Gb.NotificationHandler(
            '_gridbot.ini',
            config.getboolean("settings", "notifications"),
            config.get("settings", "notify-urls"),
        )

        # Initialise logging
        logger = Gb.Logger(
            datadir,
            '_gridbot.ini',
            notification,
            int(config.get("settings", "logrotate", fallback=7)),
            config.getboolean("settings", "debug"),
            config.getboolean("settings", "notifications"),
        )

# Initialize 3Commas API



# Auto tune a running gridbot
    while True:
        #config = Gb.load_config(datadir,program)
        #Gb.logger.info(f"Reloaded configuration from '{datadir}/{program}.ini'")

        get_last_df= san.get("ohlcv/"+symbol, 
        from_date=datetime.now()-timedelta(days=1),
        to_date=datetime.now()+timedelta(days=1),
        interval=TF).closePriceUsd

        last_data=pd.DataFrame(get_last_df)
        last_data.columns=['Price']
        conv_last_data=last_data
        conv_last_data.index = pd.to_datetime(last_data.index)

        conv_last_data.index=pd.to_datetime(conv_last_data.index).strftime(format)
        conv_last_data['ds'] = pd.to_datetime(conv_last_data.index)
        last_df_prohet=pd.DataFrame()
        last_df_no_index = conv_last_data.reset_index(drop=True)
        last_df_prohet['y']=last_df_no_index['Price']
        last_df_prohet['ds']=last_df_no_index['ds']
        
        m2 = Prophet(interval_width=0.95, 
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.5,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=0.01
            ).fit(df_prohet, init=stan_init(m))  # Adding the last day, warm-starting from m1

        future2 = m2.make_future_dataframe(periods=24*3,freq=TF)
        forecast3 = m2.predict(future2)

        last_price=round(san.get("ohlcv/"+symbol, 
            from_date=datetime.now()-timedelta(days=1),
            to_date=datetime.now(),
            interval='1m').tail(1).closePriceUsd.mean(),3)
        #last_price=round(forecast3.loc[forecast3['ds']<datetime.now(),'yhat'].tail(1).mean(),3)

        trend=forecast3.loc[forecast3['ds']>datetime.now()+timedelta(hours=24),'trend'].tail(3).mean()-\
            forecast3.loc[forecast3['ds']<datetime.now()-timedelta(hours=24),'trend'].tail(3).mean()
        
        up_lvl=round(forecast3.loc[forecast3['ds']>datetime.now()+timedelta(hours=24),'yhat_upper'].head(1).mean(),3)
        low_lvl=round(forecast3.loc[forecast3['ds']>datetime.now()+timedelta(hours=24),'yhat_lower'].head(1).mean(),3)
        
        if trend<0:    
            upper_price=up_lvl
            lower_price=last_price
        else:    
            lower_price=low_lvl
            upper_price=last_price
        
        if last_price>up_lvl:
            upper_price=up_lvl*1.1
            lower_price=last_price

        if last_price<low_lvl:
            lower_price=low_lvl*0.9
            upper_price=last_price
        #upper_price=10
        # Configuration settings
        timeint = int(Gb.config.get("settings", "timeinterval"))
        botids = json.loads(Gb.config.get("settings", "botids"))

        # Walk through all bots specified
        for bot in botids:
            boterror, botdata = Gb.api.request(
                entity="grid_bots",
                action="get",
                action_id=str(bot),
            )
            if botdata:
                #Gb.logger.debug("Raw Gridbot data: %s" % botdata)
                print('New Upper Price: ',upper_price)
                print('New Lower Price: ',lower_price)   
                Gb.manage_gridbot(botdata, upper_price, lower_price)
            else:
                Gb.logger.error("Error occurred managing gridbots: %s" % boterror["msg"])
        if not Gb.wait_time_interval(Gb.logger, Gb.notification, timeint):
            break  


