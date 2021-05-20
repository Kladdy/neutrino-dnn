class mySimulation(simulation.simulation):
    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det,
            passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
        channelBandPassFilter.run(evt, station, det,
            passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)    
    def _detector_simulation_trigger(self, evt, station, det):
        # noiseless trigger
        simpleThreshold.run(evt, station, det,
                            threshold=2.5 * self._Vrms_per_channel[station.get_id()][16],
                            triggered_channels=[16],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_2.5sigma')  # the name of the trigger
        simpleThreshold.run(evt, station, det,
                            threshold=3.5 * self._Vrms_per_channel[station.get_id()][16],
                            triggered_channels=[16],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_3.5sigma')  # the name of the trigger
        threshold_high = {}
        threshold_low = {}
        for channel_id in range(4):
            threshold_high[channel_id] = thresholds['2/4_100Hz'] * self._Vrms_per_channel[station.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_100Hz'] * self._Vrms_per_channel[station.get_id()][channel_id]
        highLowThreshold.run(evt, station, det,
                            threshold_high=threshold_high,
                            threshold_low=threshold_low,
                            coinc_window=40 * units.ns,
                            triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                            number_concidences=2,  # 2/4 majority logic
                            trigger_name='LPDA_2of4_100Hz')
        triggerTimeAdjuster.run(evt, station, det)