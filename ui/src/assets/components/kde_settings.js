
// KDE settings pane (dropdowns and sliders)

import React, { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { Button, Checkbox, Dropdown, Form, Header, Icon, Input, Ref, Segment, Select } from 'semantic-ui-react'
import LookupList from './lookup_list'
import { Slider } from 'react-semantic-ui-range'
import 'bootstrap-icons/font/bootstrap-icons.css';
import './css/filter_selection.css';
import '../../index.css';
import CONFIG from '../../../config.json';


function KDESettings({
  brightness, setBrightness, 
  numBins, setNumBins,
  kdeBrightness, setKdeBrightness,
  kdeBandwidth, setKdeBandwidth,
  kdeColormap, setKdeColormap,
  kdeColormapInvert, setKdeColormapInvert,
  kdeRelativeMode, setKdeRelativeMode,
  kdeRelativeSelection, setKdeRelativeSelection,
  kdeRowName, setKdeRowName,
  kdeColumnName, setKdeColumnName,
  facetRowValues, facetColValues }) {

  // State
  const [kdeOptions, setKdeOptions] = useState(null);

  // Num Bins Slider
  const numBinsSlider = {
    start: 0,
    min: 0,
    max: 4,
    step: 1,
    onChange: value => {
      setNumBins(value);
    }
  }

  // Brightness slider settings
  const brightnessSlider = {
    start: 1.0,
    min: 1.0,
    max: 200,
    step: 10.0,
    onChange: value => {
      setBrightness(value);
    }
  };

  // KDE bandwidth slider settings
  const kdeBandwidthSliderSettings = {
    start: kdeBandwidth,
    min: 0.0,
    max: 2.0,
    step: 0.1,
    onChange: value => {
      setKdeBandwidth(value);
    }
  };

  // KDE brightness slider settings
  const kdeBrightnessSliderSettings = {
    start: kdeBrightness,
    min: 0.0,
    max: 15.0,
    step: 0.1,
    onChange: value => {
      setKdeBrightness(value);
    }
  };

  // KDE dropdown options
  const lookupKdeOptionsUrl = CONFIG.baseUrl + 'lookup/kde/options/'
  useEffect(() => {
    fetch(lookupKdeOptionsUrl)
      .then(res => res.json())
      .then(
        (result) => {
          setKdeOptions(result)
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {
          console.log('Error loading KDE Options');
          console.log(error);
        }
      );
  }, [])


  function RenderHeatmapBinSize() {
    return (
      <React.Fragment>
        {/* Bin Count slider
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className='bi bi-grip-horizontal' />Heatmap Bin Size</Header>
        <Slider discrete
          value={numBins}
          color='blue'
          settings={numBinsSlider}
        />
      </React.Fragment>
    );
  }

  function RenderHeatmapBrightness() {
    return (
      <React.Fragment>
        {/* Brightness slider 
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className={brightness == 1.0 ? 'bi bi-lightbulb-off' : 'bi bi-lightbulb'} />Heatmap Brightness</Header>
        <Slider discrete
          value={brightness}
          color='blue'
          settings={brightnessSlider}
        />
      </React.Fragment>
    );
  }

  function RenderRelativeMode() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading relative mode options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_relative_mode'))) {
      return (
        <p>Error: kde_relative_mode' not found in KDE options.</p>
      );
    }
    let relativeModeOptions = kdeOptions['kde_relative_mode'];

    // Render dropdown
    return (
      <React.Fragment>
        <Header as='h5'><Icon className={'bi bi-filter'} />Relative Mode</Header>
        <Dropdown
          placeholder='Select relative mode...'
          fluid={false}
          selection
          options={relativeModeOptions}
          onChange={(ev, data) => setKdeRelativeMode(data.value.toString())}
          defaultValue={kdeRelativeMode}
        />
      </React.Fragment>
    );
  }

  function RenderRelativeSelection() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading relative selection options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_relative_selection'))) {
      return (
        <p>Error: kde_relative_selection' not found in KDE options.</p>
      );
    }
    let relativeSelectionOptions = kdeOptions['kde_relative_selection'];

    // Render dropdown
    return (
      <React.Fragment>
        <Header as='h5'><Icon className={'bi bi-filter'} />Relative Selection</Header>
        <Dropdown
          placeholder='Select relative selection...'
          fluid={false}
          selection
          options={relativeSelectionOptions}
          onChange={(ev, data) => setKdeRelativeSelection(data.value.toString())}
          defaultValue={kdeRelativeSelection}
        />
      </React.Fragment>
    );
  }

  function RenderColormapSelection() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading colour map options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_colormap'))) {
      return (
        <p>Error: kde_colormap' not found in KDE options.</p>
      );
    }
    let colormapOptions = kdeOptions['kde_colormap'];

    // Render dropdown
    return (
      <React.Fragment>
        <Header as='h5'><Icon className={'bi bi-filter'} />Colour Map</Header>
        <Form.Group>
          <Dropdown
            placeholder='Select colour map...'
            fluid={false}
            selection
            options={colormapOptions}
            onChange={(ev, data) => setKdeColormap(data.value.toString())}
            defaultValue={kdeColormap}
          />
          <Checkbox
            className='semantic-checkbox-adjust'
            toggle
            label='Invert'
            onChange={(ev, data) => setKdeColormapInvert(data.checked)}
            defaultChecked={kdeColormapInvert}
          />
        </Form.Group>
      </React.Fragment>
    );
  }

  function RenderRowName() {

    // Loading...
    if (facetRowValues == null || facetRowValues.length == 0 || facetRowValues[0] == '') {
      return (
        <p>No dataset loaded.</p>
      );
    }

    // Build options
    let facetRowOptions = [];
    facetRowOptions.push({ key: '', text: '', value: '' }); // blank option
    for (let i = 0; i < facetRowValues.length; i++) {
      const item = {
        key: facetRowValues[i],
        text: facetRowValues[i],
        value: facetRowValues[i],
      }
      facetRowOptions.push(item);
    }

    // Render dropdown
    return (
      <React.Fragment>
        <Header as='h5'><Icon className={'bi bi-filter'} />Row Value</Header>
        <Dropdown
          placeholder='Select row value...'
          fluid={false}
          selection
          options={facetRowOptions}
          onChange={(ev, data) => setKdeRowName(data.value.toString())}
          defaultValue={kdeRowName}
        />
      </React.Fragment>
    );
  }

  function RenderColumnName() {

    // Loading...
    if (facetColValues == null || facetColValues.length == 0 || facetColValues[0] == '') {
      return (
        <p>No dataset loaded.</p>
      );
    }

    // Build options
    let facetColOptions = [];
    facetColOptions.push({ key: '', text: '', value: '' }); // blank option
    for (let i = 0; i < facetColValues.length; i++) {
      const item = {
        key: facetColValues[i],
        text: facetColValues[i],
        value: facetColValues[i],
      }
      facetColOptions.push(item);
    }

    // Render dropdown
    return (
      <React.Fragment>
        <Header as='h5'><Icon className={'bi bi-filter'} />Column Value</Header>
        <Dropdown
          placeholder='Select column value...'
          fluid={false}
          selection
          options={facetColOptions}
          onChange={(ev, data) => setKdeColumnName(data.value.toString())}
          defaultValue={kdeColumnName}
        />
      </React.Fragment>
    );
  }

  function RenderKDESize() {
    return (
      <React.Fragment>
        {/* KDE kernel bandwidth slider
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className={'bi bi-code'} />KDE Bandwidth</Header>
        <Slider
          value={kdeBandwidth}
          color='blue'
          settings={kdeBandwidthSliderSettings}
        />
      </React.Fragment>
    );
  }

  function RenderKDEBrightness() {
    return (
      <React.Fragment>
        {/* KDE kernel brightness slider
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className={kdeBrightness == 1.0 ? 'bi bi-lightbulb-off' : 'bi bi-lightbulb'} />KDE Brightness</Header>
        <Slider
          value={kdeBrightness}
          color={'blue'}
          settings={kdeBrightnessSliderSettings}
        />
      </React.Fragment>
    );
  }

  function renderLayout() {
    return (
      <div>
        <Form size={'small'}>

          {/* Need to call as function, otherwise slider doesn't slide..? */}
          {RenderHeatmapBrightness()}
          {RenderHeatmapBinSize()}

          {/* Need to call as function, otherwise slider doesn't slide..? */}
          {RenderKDEBrightness()}
          {RenderKDESize()}
          {/*
          <RenderKDESize 
            kdeBandwidth = {kdeBandwidth}
            setKdeBandwidth = {setKdeBandwidth}
          />
          <RenderKDEBrightness
            kdeBrightness = {kdeBrightness}
            setKdeBrightness = {setKdeBrightness}
          />
          */}
          <RenderColormapSelection />
          <RenderRelativeMode />
          <RenderRelativeSelection />
          <RenderRowName />
          <RenderColumnName />
        </Form>
      </div>
    );
  }

  // Final render
  return (
    renderLayout()
  );
}

export default React.memo(KDESettings)