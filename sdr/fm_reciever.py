#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Fm Reciever
# Generated: Fri Dec 27 08:28:47 2019
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import analog
from gnuradio import audio
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio import wxgui
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.wxgui import fftsink2
from gnuradio.wxgui import forms
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import osmosdr
import time
import wx


class fm_reciever(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Fm Reciever")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.variable_slider_2 = variable_slider_2 = 50
        self.variable_slider_1 = variable_slider_1 = 50
        self.variable_slider_0 = variable_slider_0 = 100.7e6
        self.samp_rate = samp_rate = 1.0e6

        ##################################################
        # Blocks
        ##################################################
        _variable_slider_2_sizer = wx.BoxSizer(wx.VERTICAL)
        self._variable_slider_2_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_variable_slider_2_sizer,
        	value=self.variable_slider_2,
        	callback=self.set_variable_slider_2,
        	label='hp_width',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._variable_slider_2_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_variable_slider_2_sizer,
        	value=self.variable_slider_2,
        	callback=self.set_variable_slider_2,
        	minimum=10,
        	maximum=200,
        	num_steps=100,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_variable_slider_2_sizer)
        _variable_slider_1_sizer = wx.BoxSizer(wx.VERTICAL)
        self._variable_slider_1_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_variable_slider_1_sizer,
        	value=self.variable_slider_1,
        	callback=self.set_variable_slider_1,
        	label='hp_cutoff_freq',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._variable_slider_1_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_variable_slider_1_sizer,
        	value=self.variable_slider_1,
        	callback=self.set_variable_slider_1,
        	minimum=0,
        	maximum=200,
        	num_steps=100,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_variable_slider_1_sizer)
        _variable_slider_0_sizer = wx.BoxSizer(wx.VERTICAL)
        self._variable_slider_0_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_variable_slider_0_sizer,
        	value=self.variable_slider_0,
        	callback=self.set_variable_slider_0,
        	label='freq',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._variable_slider_0_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_variable_slider_0_sizer,
        	value=self.variable_slider_0,
        	callback=self.set_variable_slider_0,
        	minimum=90e6,
        	maximum=110e6,
        	num_steps=200,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_variable_slider_0_sizer)
        self.wxgui_fftsink2_1 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=0,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=250e3,
        	fft_size=2048,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot',
        	peak_hold=False,
        )
        self.Add(self.wxgui_fftsink2_1.win)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_f(
        	self.GetWin(),
        	baseband_freq=0,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=24e3,
        	fft_size=2048,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='Demod',
        	peak_hold=False,
        )
        self.Add(self.wxgui_fftsink2_0.win)
        self.rtlsdr_source_0 = osmosdr.source( args="numchan=" + str(1) + " " + '' )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(variable_slider_0, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(0, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(2, 0)
        self.rtlsdr_source_0.set_gain_mode(False, 0)
        self.rtlsdr_source_0.set_gain(10, 0)
        self.rtlsdr_source_0.set_if_gain(20, 0)
        self.rtlsdr_source_0.set_bb_gain(20, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
          
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=24,
                decimation=250,
                taps=None,
                fractional_bw=None,
        )
        self.low_pass_filter_1 = filter.fir_filter_ccf(4, firdes.low_pass(
        	1, samp_rate, 100e3, 10e3, firdes.WIN_KAISER, 6.76))
        self.high_pass_filter_0 = filter.fir_filter_fff(1, firdes.high_pass(
        	1, 24e3, variable_slider_1, variable_slider_2, firdes.WIN_HAMMING, 6.76))
        self.audio_sink_0 = audio.sink(24000, '', True)
        self.analog_wfm_rcv_0 = analog.wfm_rcv(
        	quad_rate=250e3,
        	audio_decimation=1,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_rcv_0, 0), (self.rational_resampler_xxx_0, 0))    
        self.connect((self.high_pass_filter_0, 0), (self.audio_sink_0, 0))    
        self.connect((self.high_pass_filter_0, 0), (self.wxgui_fftsink2_0, 0))    
        self.connect((self.low_pass_filter_1, 0), (self.analog_wfm_rcv_0, 0))    
        self.connect((self.low_pass_filter_1, 0), (self.wxgui_fftsink2_1, 0))    
        self.connect((self.rational_resampler_xxx_0, 0), (self.high_pass_filter_0, 0))    
        self.connect((self.rtlsdr_source_0, 0), (self.low_pass_filter_1, 0))    

    def get_variable_slider_2(self):
        return self.variable_slider_2

    def set_variable_slider_2(self, variable_slider_2):
        self.variable_slider_2 = variable_slider_2
        self._variable_slider_2_slider.set_value(self.variable_slider_2)
        self._variable_slider_2_text_box.set_value(self.variable_slider_2)
        self.high_pass_filter_0.set_taps(firdes.high_pass(1, 24e3, self.variable_slider_1, self.variable_slider_2, firdes.WIN_HAMMING, 6.76))

    def get_variable_slider_1(self):
        return self.variable_slider_1

    def set_variable_slider_1(self, variable_slider_1):
        self.variable_slider_1 = variable_slider_1
        self._variable_slider_1_slider.set_value(self.variable_slider_1)
        self._variable_slider_1_text_box.set_value(self.variable_slider_1)
        self.high_pass_filter_0.set_taps(firdes.high_pass(1, 24e3, self.variable_slider_1, self.variable_slider_2, firdes.WIN_HAMMING, 6.76))

    def get_variable_slider_0(self):
        return self.variable_slider_0

    def set_variable_slider_0(self, variable_slider_0):
        self.variable_slider_0 = variable_slider_0
        self._variable_slider_0_slider.set_value(self.variable_slider_0)
        self._variable_slider_0_text_box.set_value(self.variable_slider_0)
        self.rtlsdr_source_0.set_center_freq(self.variable_slider_0, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)
        self.low_pass_filter_1.set_taps(firdes.low_pass(1, self.samp_rate, 100e3, 10e3, firdes.WIN_KAISER, 6.76))


def main(top_block_cls=fm_reciever, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
