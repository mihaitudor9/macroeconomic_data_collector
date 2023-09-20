import time
import requests
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class IMFDataCollector:
    def __init__(self, target_countries, data_frequencies=['A', 'Q', 'M'],
                 api_url='http://dataservices.imf.org/REST/SDMX_JSON.svc/', num_workers=5):

        """
        Initializes the IMFDataCollector with target countries, data frequencies, API endpoint, and worker count.

        :param target_countries: List of countries for data collection.
        :param data_frequencies: Data frequencies, default to Annual, Quarterly, Monthly.
        :param api_url: Base URL for the IMF Data API.
        :param num_workers: Number of threads to use for concurrent data fetching.
        """

        self.target_countries = target_countries
        self.data_frequencies = data_frequencies
        self.api_url = api_url
        self.num_workers = num_workers
        self.api_session = requests.Session()
        self.indicator_descriptions = self._fetch_indicator_descriptions()

    def _fetch_indicator_descriptions(self):
        response = self.api_session.get(f'{self.api_url}DataStructure/IFS').json()
        structure = response['Structure']
        concept_schemes = structure.get('Concepts', {}).get('ConceptScheme')

        if isinstance(concept_schemes, dict):
            concept_schemes = [concept_schemes]

        if not concept_schemes:
            return {}

        indicators = next((x for x in concept_schemes if x['@id'] == 'CL_INDICATOR_IFS'), None)

        if not indicators:
            return {}

        indicator_concepts = indicators.get('Concept', [])

        if isinstance(indicator_concepts, dict):
            indicator_concepts = [indicator_concepts]

        indicator_descriptions = {item['@value']: item['Description']['#text'] for item in indicator_concepts}

        return indicator_descriptions

    def _fetch_country_data(self, country, frequency):
        api_endpoint = f'CompactData/IFS/{frequency}.{country}'
        try:
            response = self.api_session.get(f'{self.api_url}{api_endpoint}').json()
            data_series = response['CompactData']['DataSet']['Series']

            country_timeseries_list = []
            for series in data_series:
                observation_data = series.get('Obs')

                if isinstance(observation_data, dict):
                    observation_data = [observation_data]

                timeseries_dataframe = pd.DataFrame.from_dict(observation_data)
                timeseries_dataframe['@OBS_VALUE'] = timeseries_dataframe['@OBS_VALUE'].astype(float)
                timeseries_dataframe['@Country'] = series.get("@REF_AREA")
                timeseries_dataframe['@Concept'] = series.get("@INDICATOR")
                timeseries_dataframe['@ConceptDescription'] = self.indicator_descriptions.get(series.get("@INDICATOR"))
                timeseries_dataframe['@Frequency'] = series.get("@FREQ")
                country_timeseries_list.append(timeseries_dataframe)

            return country_timeseries_list
        except Exception as e:
            logging.error(f"Failed to fetch data for country {country} and frequency {frequency}. Error: {e}")

    def collect_data(self):
        start_time = time.time()
        all_dataframes = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_timeseries = {executor.submit(self._fetch_country_data, country, frequency): (country, frequency)
                                    for country in self.target_countries for frequency in self.data_frequencies}
            for future in concurrent.futures.as_completed(future_to_timeseries):
                country, frequency = future_to_timeseries[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_dataframes.extend(data)
                except Exception as e:
                    logging.error(f"An error occurred for country {country} and frequency {frequency}: {e}")

        merged_data = pd.concat(all_dataframes, axis=0)

        merged_data['@TIME_PERIOD'] = pd.to_datetime(merged_data['@TIME_PERIOD'])

        merged_data.sort_values(['@Country', '@Concept', '@Frequency', '@TIME_PERIOD'], inplace=True)

        frequency_to_period_lag = {'A': 1, 'Q': 4, 'M': 12}
        merged_data['period_lag'] = merged_data['@Frequency'].map(frequency_to_period_lag)

        merged_data['@Growth_Rate'] = merged_data.groupby(['@Country', '@Concept', 'period_lag'])[
            '@OBS_VALUE'].pct_change()

        merged_data.drop('period_lag', axis=1, inplace=True)

        logging.info("--- %s seconds ---" % (time.time() - start_time))

        return merged_data

